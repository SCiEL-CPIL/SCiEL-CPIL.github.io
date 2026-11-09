"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin using a nonlinear fragmentation-aware heuristic.
    
    Key insight: remaining space has diminishing marginal value. A bin with 10 units
    remaining is far more useful than one with 50 units, but the difference between
    50 and 100 is less critical. We use a power-law transform to create a smooth
    scoring function that naturally balances tight-fit preference with fragmentation
    avoidance, without hand-tuned piecewise thresholds.
    
    Score = -remaining^alpha, where alpha < 1 compresses large remainders and
    expands small ones, creating a nonlinear preference for tighter fits.
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
    
    rem_valid = remaining[valid_mask]
    
    # Capacity is 100 per the task description
    capacity = 100.0
    
    # Normalize remaining by capacity to make the transform scale-invariant
    rem_norm = rem_valid / capacity
    
    # Nonlinear transform: use a power law with exponent < 1
    # This compresses large remainders and expands small ones
    # alpha = 0.7: moderate compression, good balance
    alpha = 0.7
    
    # Score = -(remaining/capacity)^alpha
    # Higher score (less negative) = tighter fit preferred
    scores[valid_mask] = -(rem_norm ** alpha)
    
    # Deadline guard: if we're close to timeout, return immediately
    elapsed = time.time() - start_time
    if elapsed > 140:  # Safety margin
        return scores
    
    return scores
# EVOLVE-BLOCK-END
