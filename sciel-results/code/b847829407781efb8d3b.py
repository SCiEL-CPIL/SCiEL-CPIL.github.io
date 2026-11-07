"""
Bootstrap spawn: Direct approach using theoretical bin packing insights.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i.
    
    Strategy:
    Based on the 'Best Fit' principle in bin packing, we favor pairs of items (i, j) 
    whose combined size is as close to the bin capacity as possible. 
    This maximizes the utilization of each bin, which is the primary goal.
    
    We use a power function on the fill ratio to strongly penalize pairs that 
    leave significant empty space, effectively forcing the ACO to learn 
    high-utilization combinations first.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    # Compute pairwise sums
    # d[:, None] is (n, 1), d[None, :] is (1, n)
    # pair_sums is (n, n)
    pair_sums = d[:, None] + d[None, :]
    
    # Determine which pairs fit in a bin
    fits = pair_sums <= capacity
    
    # Calculate fill ratio for fitting pairs
    # For non-fitting pairs, we will overwrite with a small value later.
    # To avoid division by zero or log issues, we can just compute on the whole matrix 
    # and mask, or compute safely.
    
    # We want H[i,j] to be large when pair_sums[i,j] is close to capacity.
    # Let's use (pair_sums / capacity)^alpha.
    # A high alpha makes the heuristic very sensitive to the difference in fill.
    # alpha=1 is linear. alpha=10 is very strong.
    # Let's try alpha=20 to strongly drive towards "full" bins.
    
    alpha = 20.0
    beta = 15.0
    gap_weight = 0.3
    
    # Calculate heuristic values
    # For pairs that don't fit, we want 1e-9.
    # For pairs that do fit, we want a combination of fill ratio and gap-filling reward.
    
    # Initialize with the minimum valid value (1e-9)
    H = np.full((n, n), 1e-9, dtype=np.float64)
    
    # Compute values for fitting pairs
    if np.any(fits):
        ratios = pair_sums / capacity
        
        # Fill ratio term: strongly rewards pairs close to capacity
        fill_term = np.power(ratios, alpha)
        
        # Gap-filling term: rewards pairs that leave minimal unused space
        # gap = capacity - pair_sums (for fitting pairs, this is >= 0)
        # We want to reward small gaps, so use (1 - gap/capacity)^beta
        # For fitting pairs, gap/capacity is in [0, 1]
        gaps = np.maximum(capacity - pair_sums, 0) / capacity
        gap_term = np.power(1.0 - gaps, beta)
        
        # Combine: primarily fill ratio, with a smaller gap-filling bonus
        H_fit = fill_term + gap_weight * gap_term
        
        # Mask out non-fitting pairs
        H = np.where(fits, H_fit, 1e-9)
        
        # Ensure no non-finite values
        H = np.where(np.isfinite(H), H, 1e-9)
        
        # The task says "Heuristic entries that are not positive become 1e-9"
        # Our H_fit is positive for ratios > 0.
        # If any H is <= 0, set to 1e-9.
        H = np.where(H > 0, H, 1e-9)

    return H

# EVOLVE-BLOCK-END
