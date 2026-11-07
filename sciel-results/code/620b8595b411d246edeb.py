"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    # Convert to numpy array for vectorized operations
    W = np.array(demands, dtype=float)
    n = len(W)
    
    # Calculate sum of all pairs
    pair_sums = W[:, None] + W[None, :]
    
    # Calculate remaining capacity (gap) for each pair
    # If pair_sum > capacity, gap is negative (invalid)
    gaps = capacity - pair_sums
    
    # Mask for valid pairs (gap >= 0)
    fit_mask = gaps >= 0
    
    # For each pair (i, j) that fits, compute the maximum size of a third item k
    # that could still fit in the remaining space after placing i and j.
    # This is simply min(W) if the gap is large enough to fit the smallest item,
    # but we want to reward pairs that leave little or no room for a third item.
    # 
    # The "bin-completion potential" is:
    # - If gap <= 0: invalid (already handled)
    # - If gap < min(W): the pair cannot be extended with any third item -> strong reward
    # - If gap >= min(W): the pair can be extended -> lower reward
    #
    # We model this as: for each (i,j), the "closure" score is high when the gap
    # is small relative to the smallest possible third item.
    
    min_item = np.min(W)
    
    # For valid pairs, compute how "closed" the bin is after placing i and j.
    # closure = 1 / (gap + epsilon) gives high values for small gaps.
    # But we also want to penalize pairs that leave room for many more items.
    
    # A refined approach: the attractiveness should be high when the pair
    # nearly fills the bin, i.e., when gap is small.
    # We use a combination:
    #   base = 1 / (gap + epsilon)
    #   completion_bonus = 1 / (min_item + epsilon) if gap < min_item else 0
    # This gives extra reward to pairs that can't be extended.
    
    epsilon = 1e-9
    
    # Base attractiveness: inverse of gap
    safe_gaps = np.where(fit_mask, gaps + epsilon, 1.0)
    base_attr = 1.0 / safe_gaps
    
    # Completion bonus: pairs that leave less than min_item of space get a boost
    cannot_extend = fit_mask & (gaps < min_item)
    completion_bonus = np.where(cannot_extend, 1.0 / (min_item + epsilon), 0.0)
    
    # Combined heuristic
    H = np.where(fit_mask, base_attr + completion_bonus, 1e-9)
    
    # Ensure no non-finite values
    H = np.where(np.isfinite(H), H, 1e-9)
    
    # Check deadline
    if time.time() > deadline:
        # Fallback to simple size-based heuristic
        return np.where(fit_mask, W[None, :], 1e-9)
        
    return H
# EVOLVE-BLOCK-END
