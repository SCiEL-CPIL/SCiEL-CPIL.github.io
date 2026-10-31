import numpy as np
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    valid = bins >= item
    # We want the first valid bin to have the highest score.
    # If we assign scores based on index, lower index = higher score.
    # score[i] = -i if valid else -inf
    scores = np.where(valid, -np.arange(len(bins)), -np.inf)
    return scores
