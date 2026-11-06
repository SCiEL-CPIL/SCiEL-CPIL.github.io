import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid = remaining >= 0
    scores = np.full(len(bins), -np.inf)
    if np.any(valid):
        rem_valid = remaining[valid]
        bins_valid = bins[valid]
        # Best Fit base score
        scores[valid] = -rem_valid
        # Small penalty for bins that would have very little space left
        very_tight = rem_valid < 5
        if np.any(very_tight):
            idx = np.where(valid)[0][very_tight]
            scores[idx] -= 0.1 * (5 - rem_valid[very_tight])
    return scores
