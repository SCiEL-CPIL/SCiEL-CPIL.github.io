import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining capacity for each bin
    rem = bins - item
    
    # Valid bins have non-negative remaining capacity
    valid_mask = rem >= 0
    
    # Initialize scores to -inf for invalid bins
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid_mask):
        return scores
    
    valid_rem = rem[valid_mask]
    
    # Best-fit strategy: prefer bins with least remaining space
    # This packs items tightly, which is generally optimal for minimizing bin count
    # across various item size distributions including Weibull.
    valid_scores = -valid_rem
    
    scores[valid_mask] = valid_scores
    return scores
