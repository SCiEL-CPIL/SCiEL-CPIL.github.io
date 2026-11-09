import numpy as np
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Worst-Fit: prefer bins that leave the most remaining capacity.
    # Score = remaining capacity after placement.
    # Higher score = more remaining capacity.
    # We only score bins that can fit the item.
    valid = bins >= item
    scores = np.full(len(bins), -np.inf)
    scores[valid] = bins[valid] - item
    return scores
