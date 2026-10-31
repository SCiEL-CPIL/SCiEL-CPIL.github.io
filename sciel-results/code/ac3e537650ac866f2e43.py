import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid = remaining >= 0
    
    # Assign high scores to valid bins, preferring lower indices (First-Fit)
    # We want the highest score to be the lowest index.
    # So score should decrease with index.
    indices = np.arange(len(bins))
    
    # If valid, score = -index. If invalid, score = -inf.
    scores = np.where(valid, -indices, -np.inf)
    
    return scores
