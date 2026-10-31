import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining_after = bins - item
    feasible = remaining_after >= 0

    if not np.any(feasible):
        return np.full(len(bins), -np.inf)

    # First Fit: Assign highest score to the lowest index feasible bin
    indices = np.arange(len(bins))
    scores = np.full(len(bins), -np.inf)
    scores[feasible] = -indices[feasible]
    
    return scores
