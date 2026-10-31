import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Ensure bins is a numpy array
    bins = np.asarray(bins, dtype=float)
    
    # Mask invalid bins (remaining capacity < item)
    valid = bins >= item
    
    # Best-Fit: minimize remaining capacity (maximize -remaining)
    # Score = -(bins - item)
    # For invalid bins, set score to -inf
    scores = np.where(valid, -(bins - item), -np.inf)
    
    return scores
