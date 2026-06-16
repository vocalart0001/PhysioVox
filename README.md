# PhysioVox

> **An Electroglottography-Pretrained Source-Filter Network and Cross-Domain Phonation Technique Diagnosis Method for Vocal Pedagogy.**

PhysioVox is a multi-task deep-learning framework for vocal pedagogy that
treats clinical electroglottography (EGG) signals as a cross-domain
physical supervision resource. It rewrites the Liljencrants–Fant (LF)
glottal flow model into a differentiable network topology, encodes
acoustic–physiological causal correspondences as an additional loss
term, and unifies vocal-technique classification with five-dimensional
physiological parameter estimation in a single end-to-end model.

## Key Features

- **Differentiable LF glottal-flow synthesizer** — physical knowledge
  is written directly into the computation graph through Sigmoid-smoothed
  phase indication, making the classical LF model fully back-propagable.
- **EGG-driven cross-domain pretraining** — clinical voice corpora with
  synchronously recorded EGG (VOICED, SVD) are used to align the source
  representation with the physiological substrate before transfer to
  the singing domain.
- **Acoustic-physiological causal consistency loss** — pedagogical
  prior knowledge (vibrato → 5–7 Hz F0 modulation, breathy → high open
  quotient, vocal fry → high period jitter, belt → high F1) is encoded
  as an explicit objective.
- **Disentanglement regularization with MINE** — a Donsker–Varadhan
  mutual-information estimator constrains the source parameters and
  filter parameters to be statistically independent.
- **Zero-EGG inference** — at deployment only a raw waveform is needed;
  EGG is required only during pretraining.
- **Pedagogical interpretation** — the five physiological parameters are
  translated into actionable teaching language (closure, symmetry,
  impact, onset, vibrato, resonance).

## Headline Results

On VocalSet 17-class classification PhysioVox attains **Accuracy = 0.853 /
Macro-F1 = 0.838**, outperforming the strongest acoustic-only baseline
(wav2vec2-LARGE-FT, 0.776 / 0.761) by **+7.7 / +7.7 points**. The
five-dimensional physiological parameter estimation reaches **Pearson r
of 0.823 (Oq), 0.789 (Sq), 0.871 (Na), 0.768 (Rd), 0.962 (F0)** on the
held-out clinical subset.

| Dataset | Method | Accuracy | Macro-F1 |
|---|---|---|---|
| VocalSet | MPS-SVM | 0.617 | 0.591 |
| VocalSet | ResNet50-Mel | 0.694 | 0.673 |
| VocalSet | CRNN | 0.708 | 0.694 |
| VocalSet | HuBERT-Base-FT | 0.738 | 0.722 |
| VocalSet | wav2vec2-LARGE-FT | 0.776 | 0.761 |
| **VocalSet** | **PhysioVox** | **0.853** | **0.838** |
| NHSS (zero-shot) | PhysioVox | **0.628** | **0.609** |
| OpenSinger (zero-shot) | PhysioVox | **0.641** | **0.623** |

Module-wise ablation (VocalSet 17-class):

| Variant | Accuracy | Δ Accuracy |
|---|---|---|
| Full | 0.853 | — |
| w/o Differentiable LF | 0.776 | −0.077 |
| w/o EGG Pretraining | 0.802 | −0.051 |
| w/o Causal Loss | 0.819 | −0.034 |
| w/o Decoupling Regularization | 0.836 | −0.017 |
| Backbone Only | 0.741 | −0.112 |

## Repository Structure

```
PhysioVox/
├── configs/                  # YAML configurations (training + ablation)
├── data_processing/          # Five-dataset loaders and feature extraction
├── models/                   # Encoder, branches, differentiable LF, full assembly
├── losses/                   # Composite loss (L_cls, L_phys, L_caus, L_reg)
├── training/                 # Two-stage pretrain + fine-tune trainer
├── evaluation/               # Metrics, cross-domain, ablation, physiology
├── inference/                # Zero-EGG predictor + pedagogical interpreter
├── visualization/            # Figure generators (Figures 1, 6–12)
├── scripts/                  # Command-line entry points
└── results/
    ├── figures/              # Pre-rendered manuscript figures
    └── tables/               # Result tables (CSV/JSON)
```

## Installation

PhysioVox is tested on Linux and macOS with Python 3.10+.

```bash
git clone https://github.com/<your-account>/PhysioVox.git
cd PhysioVox
pip install -r requirements.txt
```

If you encounter the *externally managed environment* error on recent
Python distributions, append `--break-system-packages` to the pip command
or use a virtual environment.

## Datasets

PhysioVox is evaluated on five publicly available corpora. After
downloading each one from its official provider, run the bundled
verification script to confirm the local layout matches the structure
expected by the data loaders.

| Dataset | Subjects | Duration | EGG | Role | Source |
|---|---|---|---|---|---|
| VOICED | 208 | 0.4 h | Yes | Pretraining | https://physionet.org/content/voiced/1.0.0/ |
| SVD | 2225 sessions | 15.3 h | Yes | Pretraining | https://stimmdb.coli.uni-saarland.de/ |
| VocalSet | 20 singers | 9.8 h | No | Fine-tuning + 17-class evaluation | https://zenodo.org/record/1442513 |
| NHSS | 10 singers | 7.2 h | No | Zero-shot cross-domain test | https://hltnus.github.io/NHSSDatabase/ |
| OpenSinger | 66 singers | 48.6 h | No | Zero-shot cross-domain test | https://github.com/Multi-Singer/Multi-Singer.github.io |

Detailed download links, license terms, file-type catalogues, sample-level
metadata, and example file structures (including real `voice001.hea`,
`voice001-info.txt`, `Song.TextGrid` Praat tier samples, `.lab` HTK
phoneme alignment samples, and `readme-anon.txt` listings) are provided
in the companion materials archive (see *Companion Materials* below).

## Quick Start

### Reproduce Pre-Computed Figures

```bash
python scripts/reproduce_figures.py
```

This regenerates Figures 1, 6, 7, 8, 9, 10, 11, 12 of the manuscript and
saves them as 300-DPI PNGs under `results/figures/`.

### Verify the Dataset Layout

```bash
python verify_datasets.py \
    --voiced     /path/to/VOICED/voiced-1.0.0 \
    --svd        /path/to/SVD \
    --vocalset   /path/to/VocalSet/VocalSet1-2 \
    --nhss       /path/to/NHSS_Database \
    --opensinger /path/to/OpenSinger
```

### Full Two-Stage Training Pipeline

```bash
# Stage 1: EGG-driven cross-domain pretraining on VOICED + SVD
python scripts/run_pretrain.py \
    --config configs/default.yaml \
    --voiced /path/to/VOICED \
    --svd    /path/to/SVD

# Stage 2: VocalSet fine-tuning with parameter-distance regularisation
python scripts/run_finetune.py \
    --config configs/default.yaml \
    --vocalset   /path/to/VocalSet \
    --pretrained results/checkpoints/pretrain_best.pt

# End-to-end evaluation (VocalSet + cross-domain + physiological)
python scripts/run_evaluation.py \
    --config     configs/default.yaml \
    --checkpoint results/checkpoints/finetune_best.pt \
    --vocalset   /path/to/VocalSet \
    --nhss       /path/to/NHSS \
    --opensinger /path/to/OpenSinger \
    --voiced     /path/to/VOICED \
    --svd        /path/to/SVD

# Module ablation and loss-weight sensitivity sweep
python scripts/run_ablation.py \
    --config   configs/default.yaml \
    --ablation configs/ablation.yaml \
    --vocalset /path/to/VocalSet \
    --pretrained results/checkpoints/pretrain_best.pt
```

### Zero-EGG Inference and Pedagogical Interpretation

```python
from inference import PhysioVoxPredictor, PedagogicalInterpreter
import yaml

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)

predictor = PhysioVoxPredictor(
    config=config,
    checkpoint_path="results/checkpoints/finetune_best.pt",
)
output = predictor.predict("my_singing_sample.wav")
print(output["technique"])       # e.g. "vibrato"
print(output["psi"])              # {Oq, Sq, Na, Rd, F0}

interpreter = PedagogicalInterpreter()
guidance = interpreter.interpret(
    psi=output["psi"],
    formants=output["formants"],
    f0_modulation_rate=5.8,
    f0_modulation_depth=0.062,
    period_jitter=0.012,
)
print(guidance["recommendation"])
```

## Method Summary

PhysioVox is a five-layer pipeline:

1. **Input** — raw waveform resampled to 16 kHz.
2. **Acoustic features** — 80-dimensional Mel spectrogram with 25 ms
   window and 10 ms hop.
3. **Shared encoder** — 6-layer Transformer with d_model = 512 and h = 8
   attention heads.
4. **Two branches**
   - *Glottal source branch* — produces the five physiologically
     interpretable LF parameters {Oq, Sq, Na, Rd, F0} via a Sigmoid
     reparameterization to the feasible region.
   - *Vocal-tract filter branch* — produces LPC-12 reflection
     coefficients (mapped to direct-form coefficients through the
     Levinson–Durbin step-up recursion) and four formant frequencies.
5. **Differentiable LF synthesizer + classifier** — the source parameters
   drive a closed-form glottal-flow waveform; the LPC filter shapes that
   waveform into a reconstructed speech signal; the pooled encoder
   latent feeds a 17-class technique classifier.

The composite loss has four components:

```
L_total = L_cls + λ_phys · L_phys + λ_caus · L_caus + λ_reg · L_reg
```

- **L_cls** — cross-entropy with label smoothing for 17-class technique
  classification.
- **L_phys** — time-domain MSE plus multi-scale log-spectral distance
  between the synthesized glottal flow and the EGG ground truth.
- **L_caus** — four-term causal consistency objective covering vibrato,
  breathy, vocal fry, and belt techniques.
- **L_reg** — log-magnitude spectral smoothness on the source spectrum
  plus a MINE-based mutual-information penalty between source and filter
  parameters.

The optimal operating point is **λ_phys = 1.0**, **λ_caus = 0.5**,
**λ_reg = 0.1**.

## Companion Materials

A separate materials archive contains the dataset metadata workbook
(twelve sheets covering subject-level metadata, file-type catalogues,
and example file structures for all five corpora) together with the
download-links document and the layout-verification script. Place it
alongside this repository or upload it to your project's *Releases* tab.

## Citation

If you build on PhysioVox in your research, please cite the corresponding
manuscript.

```bibtex
@article{physiovox2026,
  title   = {An Electroglottography-Pretrained Source-Filter Network and
             Cross-Domain Phonation Technique Diagnosis Method for Vocal
             Pedagogy},
  year    = {2026},
}
```

## License

The PhysioVox source code is released under the MIT License. The five
external datasets are governed by their respective providers' licenses;
please consult each dataset's official documentation before
redistribution.
