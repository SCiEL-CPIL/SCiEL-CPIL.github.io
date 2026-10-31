import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    C = 100.0
    R = bins
    
    # Initialize scores to a very low value for invalid bins
    scores = np.full_like(R, -1e9, dtype=float)
    
    # Identify valid bins
    valid = R >= item
    if not np.any(valid):
        return scores
        
    valid_R = R[valid]
    
    # Threshold for switching strategy
    # Weibull items are often small.
    # If item is small, consolidate (Best-Fit).
    # If item is large, spread (Worst-Fit) to preserve flexibility.
    if item < C * 0.5:
        # Best-Fit: Minimize remaining capacity
        scores[valid] = -valid_R
    else:
        # Worst-Fit: Maximize remaining capacity
        scores[valid] = valid_R
        
    return scores
