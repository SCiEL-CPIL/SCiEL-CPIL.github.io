"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Weibull-distributed items with capacity 100: typical median is around 20-30
# Use a proxy for median item size to score "next-item fit"
_MEDIAN_ITEM_PROXY = 25.0  # Approximate median of Weibull items with cap 100

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin using a smooth nonlinear Best Fit heuristic with Next-Item Fit proxy.
    
    Uses a power-law transformation on normalized remaining capacity combined with
    a penalty for remaining capacity that deviates from the median item size,
    encouraging bins that can accommodate typical future items.
    """
    start_time = time.time()
    
    # Compute remaining capacity if item is placed in each bin
    remaining = bins - item
    
    # Only consider bins with enough remaining capacity
    valid_mask = remaining >= 0
    
    # Create scores array, default to very low for invalid bins
    scores = np.full(len(bins), -np.inf, dtype=np.float64)
    
    if not np.any(valid_mask):
        return scores
    
    # Normalize remaining capacity by bin capacity (bins are all capacity 100)
    capacity = 100.0
    norm_remaining = remaining[valid_mask] / capacity
    
    # Smooth nonlinear scoring: power law for tight fits
    alpha = 1.5
    base_score = -np.power(norm_remaining, alpha)
    
    # Next-Item Fit proxy: penalize deviation from median item size
    # We want remaining capacity close to median item size for future items
    # This is a smooth surrogate for the nonsmooth min/max of fragmentation
    median_proxy = _MEDIAN_ITEM_PROXY / capacity
    deviation = np.abs(norm_remaining - median_proxy)
    
    # Combine: base best-fit score with a smooth penalty for poor next-item fit
    # The penalty is weighted to be secondary to the primary best-fit objective
    penalty_weight = 0.3
    next_item_penalty = penalty_weight * deviation
    
    scores[valid_mask] = base_score - next_item_penalty
    
    # Deadline guard: if we're close to timeout, return immediately
    elapsed = time.time() - start_time
    if elapsed > 140:  # Safety margin
        return scores
    
    return scores
# EVOLVE-BLOCK-END
