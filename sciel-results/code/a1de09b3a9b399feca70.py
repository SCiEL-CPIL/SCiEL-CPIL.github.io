"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Hybrid Best/Worst Fit based on item size relative to bin capacity.
    - For small items (item < 0.3 * capacity): use Best Fit (minimize remaining space)
      to consolidate small items and reduce fragmentation.
    - For large items (item >= 0.3 * capacity): use Worst Fit (maximize remaining space)
      to keep bins open for other large items.
    
    We estimate capacity as the max of bins (assuming bins represent remaining capacity
    and full bins are 0, so max is close to original capacity). Actually, in online bin packing
    with remaining capacity, bins start at capacity and decrease. So max(bins) approximates
    the capacity of the least-used bin, which is a good proxy for bin capacity.
    """
    # Determine which bins have enough remaining capacity
    valid_mask = bins >= item
    if not np.any(valid_mask):
        # No valid bins - return negative infinity for all (shouldn't happen in valid eval)
        return -np.inf * np.ones_like(bins, dtype=np.float64)
    
    # Estimate bin capacity: use max remaining capacity as proxy
    cap = np.max(bins) if len(bins) > 0 else 100.0
    
    # Threshold for switching between best-fit and worst-fit
    threshold = 0.3 * cap
    
    if item < threshold:
        # Best Fit: prefer bins with least remaining capacity after placing item
        # Score = -(remaining after placement) so higher score = tighter fit
        remaining = bins - item
        scores = -remaining
    else:
        # Worst Fit: prefer bins with most remaining capacity after placing item
        # Score = remaining after placement so higher score = more room left
        remaining = bins - item
        scores = remaining
    
    # Set invalid bins to very low score so they're never chosen
    scores = np.where(valid_mask, scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
