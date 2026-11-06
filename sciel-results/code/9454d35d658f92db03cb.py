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
    
    # Size-adaptive strategy:
    # - Small items (< 25): prefer bins with more remaining space (worst-fit)
    #   to spread them out and leave room for larger items
    # - Large items (>= 25): prefer bins with least remaining space (best-fit)
    #   to pack them tightly
    if item < 25.0:
        # Worst-fit: maximize remaining capacity
        valid_scores = valid_rem
    else:
        # Best-fit: minimize remaining capacity
        valid_scores = -valid_rem
    
    scores[valid_mask] = valid_scores
    return scores
