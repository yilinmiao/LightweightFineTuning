---
base_model: gpt2
library_name: peft
---

# Model Card for Model ID

<!-- Provide a quick summary of what the model is/does. -->



## Model Details

### Model Description

This project demonstrates lightweight fine-tuning of a pre-trained language model using Parameter-Efficient Fine-Tuning (PEFT) techniques.

- **Developed by:** Ian Miao
- **Model type:** GPT-2 with Low-Rank Adaptation (LoRA)
- **Language:** English
- **License:** MIT
- **Finetuned from model:** GPT-2 (base model from Hugging Face)

### Model Sources

- **Base Model:** [GPT-2](https://huggingface.co/gpt2)
- **PEFT Library:** [PEFT](https://github.com/huggingface/peft)

## Project Overview

This project implements a lightweight fine-tuning approach using Low-Rank Adaptation (LoRA) on the GPT-2 model for sentiment analysis. The Stanford Sentiment Treebank (SST-2) dataset is used for fine-tuning and evaluation.

### Key Components

- **PEFT Technique:** Low-Rank Adaptation (LoRA)
- **Base Model:** GPT-2 (gpt2)
- **Task:** Sentiment Analysis
- **Dataset:** Stanford Sentiment Treebank (SST-2)
- **Evaluation Approach:** Accuracy metric with Hugging Face's Trainer

## How to Get Started with the Model

The implementation is available in the Jupyter notebook `LightweightFineTuning.ipynb`. To run the notebook:

1. Ensure you have the required dependencies installed:
   ```
   pip install transformers datasets evaluate peft torch accelerate
   ```

2. Open and run the notebook to see the implementation details, training process, and evaluation results.

## Training Details

### Training Data

The model is fine-tuned on the Stanford Sentiment Treebank (SST-2) dataset, which contains movie reviews labeled with binary sentiment (positive/negative).

### Training Procedure

The training uses LoRA, which adds trainable rank decomposition matrices to existing weights while freezing the original model parameters. This significantly reduces the number of trainable parameters compared to full fine-tuning.

#### Training Hyperparameters

- **PEFT Method:** LoRA
- **Rank:** 8
- **Alpha:** 16
- **Dropout:** 0.1
- **Training Regime:** Mixed precision (fp16)

## Evaluation

### Metrics

The model is evaluated using accuracy on the SST-2 validation set.

### Results

The fine-tuned model demonstrates improved performance on sentiment analysis compared to the base model, while requiring significantly fewer trainable parameters.

## Technical Specifications

### Compute Infrastructure

- The model was trained using PyTorch and the Hugging Face Transformers and PEFT libraries.

## Framework Versions

- PEFT 0.14.0
- Transformers (Hugging Face)
- PyTorch
- Datasets (Hugging Face)

## Model Card for Model ID

<!-- Provide a quick summary of what the model is/does. -->



## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->



- **Developed by:** [More Information Needed]
- **Funded by [optional]:** [More Information Needed]
- **Shared by [optional]:** [More Information Needed]
- **Model type:** [More Information Needed]
- **Language(s) (NLP):** [More Information Needed]
- **License:** [More Information Needed]
- **Finetuned from model [optional]:** [More Information Needed]

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** [More Information Needed]
- **Paper [optional]:** [More Information Needed]
- **Demo [optional]:** [More Information Needed]

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

[More Information Needed]

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

[More Information Needed]

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

[More Information Needed]

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

[More Information Needed]

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.

## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

[More Information Needed]

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** [More Information Needed]
- **Hours used:** [More Information Needed]
- **Cloud Provider:** [More Information Needed]
- **Compute Region:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]

## Technical Specifications [optional]

### Model Architecture and Objective

[More Information Needed]

### Compute Infrastructure

[More Information Needed]

#### Hardware

[More Information Needed]

#### Software

[More Information Needed]

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

[More Information Needed]

**APA:**

[More Information Needed]

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

[More Information Needed]

## More Information [optional]

[More Information Needed]

## Model Card Authors [optional]

[More Information Needed]

## Model Card Contact

[More Information Needed]