import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Best Fit: minimize remaining space (bins - item)
    # Equivalent to maximizing item - bins? No.
    # We want the bin where (bins - item) is smallest but >= 0.
    # So we want to maximize -(bins - item) = item - bins.
    
    # Calculate remaining space if item is placed
    remaining = bins - item
    
    # Score should be high for valid bins with small remaining space.
    # Score = -remaining
    # For invalid bins (remaining < 0), score should be very low.
    
    scores = -remaining
    # Mask invalid bins
    invalid = remaining < 0
    scores[invalid] = -np.inf
    
    return scores
