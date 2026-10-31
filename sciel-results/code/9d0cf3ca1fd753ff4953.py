import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid_mask = remaining >= 0
    
    if not np.any(valid_mask):
        return np.full_like(bins, -np.inf)
    
    scores = np.full_like(bins, -np.inf)
    
    # Hard threshold at 50% of capacity
    # Capacity is 100
    if item > 50.0:
        # Worst Fit: Maximize remaining
        scores[valid_mask] = remaining[valid_mask]
    else:
        # Best Fit: Minimize remaining
        scores[valid_mask] = -remaining[valid_mask]
        
    return scores
