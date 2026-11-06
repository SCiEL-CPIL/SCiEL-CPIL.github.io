"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for placing an item. Higher score = better bin.
    Uses a smooth best-fit variant that accounts for item size relative to capacity.
    """
    # Remaining capacity in each bin
    remaining = bins - item
    
    # Only consider bins with enough capacity
    valid_mask = remaining >= 0
    
    if not np.any(valid_mask):
        # No valid bin - return very low scores
        return np.full_like(bins, -1e10, dtype=np.float64)
    
    # For valid bins, we want to minimize remaining capacity (best-fit)
    # But with a smooth adaptation: for small items, prefer bins with more remaining
    # to keep large bins available; for large items, use best-fit tightly.
    
    # Normalize remaining capacity to [0, 1] range for valid bins
    # Use a smooth score that penalizes leaving too much space for large items
    # but allows some flexibility for small items
    
    # Item size ratio (assuming capacity ~100 based on task description)
    # We don't know exact capacity, but we can infer from max bin value
    cap_estimate = np.max(bins[valid_mask]) + item  # rough estimate
    
    # For best-fit: score = -remaining (lower remaining = higher score)
    # Smooth adaptation: add a small bonus for bins that are "just right"
    # Use a Gaussian-like preference centered at small remaining values
    
    # Base best-fit score
    base_score = -remaining.copy()
    
    # Adaptive component: for items < 20% of capacity, slightly prefer
    # bins with more remaining to preserve flexibility
    # For items > 20%, stick closer to pure best-fit
    if item < 0.2 * cap_estimate:
        # Add a small positive term for remaining capacity (prefer more room)
        # This creates a balance between best-fit and worst-fit
        adaptive = 0.1 * remaining
        base_score[valid_mask] += adaptive[valid_mask]
    else:
        # For larger items, pure best-fit is better
        # Add a small penalty for bins that are too full (to avoid fragmentation)
        pass
    
    # Invalidate bins that don't fit
    base_score[~valid_mask] = -1e10
    
    return base_score
# EVOLVE-BLOCK-END
