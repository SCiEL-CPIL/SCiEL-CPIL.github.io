import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining space
    remaining = bins - item
    
    # We only care about bins that can fit the item
    # For bins that cannot fit, we assign a very low score (or -inf)
    # But the interface says "score each bin with enough remaining capacity".
    # It implies we return scores for all bins, but the caller picks the highest among valid ones?
    # "each arriving item goes to the highest-scoring bin"
    # If a bin cannot fit, it should not be chosen.
    # So we should give invalid bins a very low score.
    
    valid = remaining >= 0
    scores = np.full_like(remaining, -np.inf)
    
    if np.any(valid):
        # Use inverse of remaining space to strongly prefer tight fits
        # Add epsilon to avoid division by zero
        eps = 1e-9
        scores[valid] = -1.0 / (remaining[valid] + eps)
        
    return scores
