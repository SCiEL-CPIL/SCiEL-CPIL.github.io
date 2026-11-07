"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    pair_sum = demands[:, None] + demands[None, :]
    gap = capacity - pair_sum
    
    # Initialize with very small value for invalid pairs
    H = np.full((n, n), 1e-9)
    
    # Valid pairs are those where both items fit together
    valid = gap >= 0
    
    # For valid pairs, reward tight packing (small gap)
    # Use inverse of (gap + 1) to avoid division by zero
    # This rewards pairs that leave minimal residual space
    H[valid] = 1.0 / (gap[valid] + 1.0)
    
    # Additionally, strongly penalize pairs that create dead gaps
    # (residual space too small to fit any remaining item)
    # A dead gap means the bin is "wasted" for future items
    min_demand = np.min(demands)
    dead_gap = valid & (gap > 0) & (gap < min_demand)
    
    # For dead gaps, we want to DISCOURAGE these pairs
    # But wait - the task says "larger is more attractive"
    # So we should make dead gaps LESS attractive
    # The inverse gap already makes smaller gaps more attractive
    # But dead gaps (small but not zero) are bad because they can't be used
    
    # Let's refine: give a boost to pairs that either:
    # 1. Fill the bin exactly (gap == 0) - best case
    # 2. Leave a gap that can still accommodate another item (gap >= min_demand)
    # And penalize dead gaps (0 < gap < min_demand)
    
    # Reset and rebuild with better logic
    H = np.full((n, n), 1e-9)
    
    # Exact fit (gap == 0): highest attractiveness
    exact_fit = valid & (gap == 0)
    H[exact_fit] = 1000.0
    
    # Good gaps (gap >= min_demand): can still add another item
    good_gap = valid & (gap >= min_demand)
    # Use inverse of gap to prefer smaller good gaps
    H[good_gap] = 1.0 / (gap[good_gap] + 1.0)
    
    # Dead gaps (0 < gap < min_demand): penalize
    dead_gap_mask = valid & (gap > 0) & (gap < min_demand)
    # Give them low attractiveness but not the minimum
    H[dead_gap_mask] = 0.01
    
    return H
# EVOLVE-BLOCK-END
