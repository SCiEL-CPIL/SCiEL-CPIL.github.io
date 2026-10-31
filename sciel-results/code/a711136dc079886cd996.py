"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for deadline
_start_time = time.time()
_DEADLINE = _start_time + 140.0  # safety margin before 150s kill

# Golden ratio target for remainder
_PHI = 0.6180339887498949
_CAPACITY = 100.0
_TARGET_REM = _PHI * _CAPACITY  # ~61.8

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Uses a hybrid of:
    1. Best-fit-decreasing style: prefer bins where remainder is close to golden ratio target
    2. Penalize very small remainders (fragmentation)
    3. Slight preference for bins that are "fuller" (higher utilization)
    
    Returns an array of scores (higher is better for selection).
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback to simple best-fit
        rem = bins - item
        valid = rem >= 0
        scores = np.full_like(bins, -np.inf, dtype=float)
        scores[valid] = -rem[valid]
        return scores
    
    rem = bins - item
    valid = rem >= 0
    
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    if not np.any(valid):
        return scores
    
    rem_v = rem[valid]
    
    # Component 1: Distance to golden ratio target (smaller distance = better)
    # We want remainders near _TARGET_REM
    dist_to_target = np.abs(rem_v - _TARGET_REM)
    
    # Component 2: Penalty for very small remainders (< 5% of capacity)
    small_rem_penalty = np.where(rem_v < 5.0, (5.0 - rem_v) * 2.0, 0.0)
    
    # Component 3: Mild preference for higher utilization (fuller bins)
    # This is already partially captured by best-fit, but let's add a small term
    # Actually, best-fit already prefers smaller remainders. The golden ratio target
    # may conflict with pure best-fit. Let's blend.
    
    # Normalize dist_to_target to a reasonable range
    # Max possible dist is max(_TARGET_REM, _CAPACITY - item) - min(rem_v)
    # Simpler: just use the raw distance, scaled down
    score_v = -dist_to_target - small_rem_penalty
    
    # Add a small bonus for remainders that are "useful" (between 10 and 90)
    useful_bonus = np.where((rem_v >= 10.0) & (rem_v <= 90.0), 0.5, 0.0)
    score_v += useful_bonus
    
    scores[valid] = score_v
    return scores
# EVOLVE-BLOCK-END
