"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0  # seconds before the 150s limit
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the arriving item.
    
    Strategy: Fluid Flow / Relative Waste Minimization.
    Instead of just minimizing absolute remainder (Best Fit), we prioritize bins
    where the item constitutes a large fraction of the remaining capacity.
    
    The score is based on the ratio: item / remaining_capacity.
    Higher ratio means the item fills a larger portion of the available space,
    which reduces relative waste and helps consolidate items into fewer bins.
    
    We also add a small penalty for very small remainders to avoid creating
    unusable gaps that can't fit future items (based on observed improvements).
    
    For infeasible bins (remaining < 0), score is -inf.
    """
    # Check if we're close to the deadline
    if time.time() - _START_TIME > _MAX_EVAL_TIME:
        # Emergency: return a simple best-fit fallback
        remaining = bins - item
        scores = np.where(remaining >= 0, -remaining, -np.inf)
        return scores
    
    remaining = bins - item
    
    # Feasible bins: remaining >= 0
    feasible = remaining >= 0
    
    # Initialize scores to -inf
    scores = np.full(len(bins), -np.inf, dtype=float)
    
    if not np.any(feasible):
        return scores
    
    # For feasible bins, compute the ratio of item to remaining capacity
    # This represents how "well" the item fits in terms of relative fill
    # We want to maximize this ratio (higher is better)
    
    # Avoid division by zero: add a small epsilon to remaining
    eps = 1e-9
    valid_remaining = remaining[feasible] + eps
    
    # Base score: ratio of item to remaining (higher is better)
    # This is the "fluid flow" heuristic
    ratio = item / valid_remaining
    
    # Additional refinement: penalize very small remainders
    # If the remaining capacity after placing the item is very small (< 5),
    # it might create an unusable gap. We slightly penalize these cases.
    # However, we don't want to over-penalize because sometimes a tight fit is necessary.
    # Let's add a small bonus for moderate remainders and a small penalty for tiny ones.
    
    # Penalty for very small remainders (less than 5 units)
    # This is a soft penalty to avoid fragmentation
    small_remainder_mask = remaining[feasible] < 5.0
    penalty = np.zeros_like(ratio)
    penalty[small_remainder_mask] = 0.1  # Small penalty
    
    # Final score: ratio minus penalty
    # We want to maximize this, so higher score is better
    scores[feasible] = ratio - penalty
    
    return scores
# EVOLVE-BLOCK-END
