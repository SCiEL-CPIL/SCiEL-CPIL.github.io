import numpy as np
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    capacity = 100.0 # Known from task description
    remaining = bins - item
    valid = remaining >= 0
    
    if not np.any(valid):
        return np.full(len(bins), -np.inf)
    
    valid_remaining = remaining[valid]
    min_remaining = np.min(valid_remaining)
    
    # Threshold: if the best fit is very tight, use Best-Fit.
    # Otherwise, use Worst-Fit to balance?
    # Let's try a threshold of 0.1 * capacity = 10.0
    threshold = 10.0
    
    scores = np.full(len(bins), -np.inf)
    
    if min_remaining < threshold:
        # Best-Fit: maximize -remaining (minimize remaining)
        scores[valid] = -valid_remaining
    else:
        # Worst-Fit: maximize remaining
        scores[valid] = valid_remaining
        
    return scores
