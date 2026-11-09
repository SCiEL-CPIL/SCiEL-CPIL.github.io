"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_deadline = time.time() + 140  # Safety margin before 150s kill

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the item.
    The item goes to the highest-scoring bin.
    
    Strategy: Hybrid Best-Fit/Worst-Fit with nonlinear fragmentation penalty.
    
    Key insights:
    1. Pure Best-Fit minimizes immediate waste but can create many small unusable gaps.
    2. Pure Worst-Fit preserves large spaces but wastes capacity on small items.
    3. For Weibull (skewed) distributions, we need to balance:
       - Prefer tight fits (Best-Fit) to maximize density.
       - Avoid creating tiny gaps that will never be filled (fragmentation penalty).
       - Slightly prefer bins with more remaining space if the tight fit creates a "dead" gap.
    
    Scoring formula:
    - Base: -waste^alpha (Nonlinear Best-Fit component, alpha < 1 compresses large remainders)
    - Bonus: beta * bins[i] (Worst-Fit component, preserves capacity)
    - Penalty: gamma * waste * (1 - waste/item) for small gaps (waste < item)
    
    The penalty term is designed to discourage creating gaps smaller than the current item,
    as these are likely unusable for future items of similar scale.
    """
    remaining = bins - item
    valid = remaining >= 0
    
    scores = np.full(len(bins), -np.inf, dtype=np.float64)
    if not np.any(valid):
        return scores
        
    waste = remaining[valid]
    bin_rem = bins[valid]
    
    # Capacity is 100 per the task description
    capacity = 100.0
    
    # Normalize remaining by capacity to make the transform scale-invariant
    rem_norm = waste / capacity
    
    # Base score: Nonlinear Best-Fit (minimize waste with diminishing returns)
    # alpha = 0.7: moderate compression, good balance
    alpha = 0.7
    base_score = -(rem_norm ** alpha)
    
    # Worst-Fit component: slightly prefer bins with more remaining capacity
    # This helps preserve large contiguous space for future large items
    # We use a small coefficient to not override the Best-Fit signal
    worst_fit_bonus = 0.05 * bin_rem
    
    # Fragmentation penalty: penalize creating small gaps that are likely unusable
    # If waste is small relative to the item, it's probably dead space
    # We use a smooth penalty that is 0 when waste >= item and increases as waste -> 0
    # Penalty = gamma * waste * (1 - waste/item) for waste < item
    
    # To avoid division by zero, we add a small epsilon
    item_safe = item + 1e-9
    gap_ratio = waste / item_safe
    # Penalty is active when gap_ratio < 1 (waste < item)
    # We use a smooth transition
    penalty_mask = gap_ratio < 1.0
    penalty = np.zeros_like(waste)
    if np.any(penalty_mask):
        # For small gaps, penalize based on how small they are
        # A gap of 0 is worst, a gap of item is fine
        # Penalty = gamma * waste * (1 - gap_ratio)
        # This gives 0 penalty at waste=item and gamma*waste penalty at waste=0
        # But waste=0 is actually perfect (exact fit), so we need to be careful
        # Actually, if waste is very small, it's "dead space" unless it's exactly 0
        # Let's use a penalty that peaks at some intermediate small value
        # Or simply: if waste < threshold, add a penalty
        # Let's use: penalty = gamma * waste * (1 - gap_ratio) for gap_ratio < 1
        # This encourages either exact fits (waste~0) or larger gaps (waste>item)
        # Wait, if waste is 0, penalty is 0. If waste is 0.5*item, penalty is gamma*0.5*item*0.5 = 0.25*gamma*item
        # If waste is 1.5*item, penalty is 0.
        # This seems reasonable.
        penalty[penalty_mask] = 0.1 * waste[penalty_mask] * (1 - gap_ratio[penalty_mask])
    
    scores[valid] = base_score + worst_fit_bonus - penalty
    
    # Deadline guard: if we're close to timeout, return immediately
    elapsed = time.time() - _deadline
    if elapsed > 140:  # Safety margin
        return scores
    
    return scores

# EVOLVE-BLOCK-END
