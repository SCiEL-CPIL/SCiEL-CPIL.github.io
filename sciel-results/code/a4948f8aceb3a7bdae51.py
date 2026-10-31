# EVOLVE-BLOCK-START
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Use effective capacity scoring where we target remaining space
    to be close to a fraction of the current item size, balancing tight packing
    with the ability to fit future items.
    
    score = -(remaining - target) where target = item * 0.3
    This prefers bins where remaining ≈ 30% of current item size.
    """
    # Initialize scores to -infinity
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    # Find valid bins: those with enough remaining capacity for the current item
    valid_mask = bins >= item
    
    if not np.any(valid_mask):
        # No valid bins; return all -inf
        return scores
    
    # Calculate remaining space for valid bins
    remaining = bins[valid_mask] - item
    
    # Target remaining space: a fraction of the current item size
    # Using 0.3 as a balance between tight packing and future item fit
    target = item * 0.3
    
    # Score: negative of (remaining - target)
    # Higher score when remaining is close to target
    scores[valid_mask] = -(remaining - target)
    
    return scores
# EVOLVE-BLOCK-END
