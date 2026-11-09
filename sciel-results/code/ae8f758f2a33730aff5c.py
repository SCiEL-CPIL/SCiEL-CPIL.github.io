"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We'll use a more sophisticated scoring that considers:
# 1. How full the bin will be after placing the item (higher is better)
# 2. A penalty for leaving small gaps that can't be filled by future items
# 3. A bonus for bins that are "balanced" - not too empty, not too full
# 
# The key insight: we want to pack items densely, but we also want to avoid
# creating many small fragments. A good heuristic is to prefer bins where
# the item fits well (high utilization) but also consider the distribution
# of remaining space.

# Precompute some constants for speed
_CAP = 100.0  # Bin capacity (from problem description)

# For Weibull-distributed items with capacity 100, typical item sizes range
# from small to moderate. We'll use a scoring function that:
# - Rewards high utilization after placement
# - Penalizes very small remaining spaces (wasted capacity)
# - Has a smooth transition to avoid discontinuities

def _score_bins(bins: np.ndarray, item: float) -> np.ndarray:
    """Score bins based on how well the item fits."""
    remaining = bins - item
    # Only consider bins with enough capacity
    valid = remaining >= 0
    
    # Utilization after placement: how full the bin becomes
    utilization = (_CAP - remaining) / _CAP  # = item + used_capacity / cap
    
    # We want high utilization, but also want to avoid tiny remaining spaces
    # that are too small to be useful for future items.
    
    # A good approach: score = utilization * (1 - penalty_for_small_remainder)
    # where penalty increases as remainder gets very small (but not zero)
    
    # Let's use a sigmoid-based penalty:
    # - If remaining is close to 0, that's actually good (perfect fit)
    # - If remaining is very small but not zero, it's bad (wasted space)
    # - If remaining is moderate, it's okay
    # - If remaining is large, it's less ideal (bin is underutilized)
    
    # Better approach: use the concept of "effective utilization"
    # Score higher when the item fills the bin well, but also consider
    # that leaving some flexible space is valuable.
    
    # Let's try: score = -remaining + bonus_for_good_fit
    # where bonus rewards bins where remaining is either ~0 or in a "good" range
    
    # Actually, let's think about this differently. The L1 lower bound is
    # sum(items) / capacity. We want to minimize the number of bins used.
    # So we should prefer placements that keep bins as full as possible.
    
    # Simple effective score: prefer bins where after placement, the bin
    # is as full as possible, but with a slight preference for bins that
    # have a "clean" remaining space.
    
    # Use: score = -remaining * (1 + alpha * (remaining < threshold))
    # This penalizes bins with very small remaining spaces more.
    
    # Let's use a smooth function:
    # score = -remaining + beta * exp(-gamma * remaining)
    # The exponential term gives a bonus for bins that are nearly full.
    
    beta = 2.0
    gamma = 0.1
    
    scores = -remaining + beta * np.exp(-gamma * remaining)
    scores[~valid] = -np.inf
    
    return scores


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Score each bin with enough remaining capacity for the item.
    
    Args:
        item: The size of the arriving item.
        bins: Array of remaining capacities for each bin.
        
    Returns:
        Array of scores, one per bin. Higher score = better choice.
        Bins that cannot fit the item get score -inf.
    """
    start_time = time.time()
    
    # Ensure bins is a numpy array
    bins = np.asarray(bins, dtype=np.float64)
    
    # Compute scores using our heuristic
    scores = _score_bins(bins, item)
    
    # Safety margin: if we're taking too long, return immediately
    elapsed = time.time() - start_time
    if elapsed > 0.1:  # 100ms safety margin
        return scores
    
    return scores
# EVOLVE-BLOCK-END
