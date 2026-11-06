"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_START_TIME = time.time()
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 5.0  # seconds before deadline to stop


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Returns an array of scores; the item goes to the highest-scoring bin.
    
    Strategy: Best-Fit with adaptive fragment penalty.
    - Base score: -remaining (prefer bins with least remaining capacity)
    - Penalty: smooth penalty for creating very small residuals (fragments)
      that are unlikely to be useful for future items
    - The penalty threshold scales with item size and bin capacity
    """
    # Deadline guard: if we're running out of time, fall back to simple best-fit
    if time.time() > _START_TIME + DEADLINE - SAFETY_MARGIN:
        remaining = bins - item
        valid_mask = remaining >= -1e-12
        scores = np.full(len(bins), -np.inf)
        if np.any(valid_mask):
            scores[valid_mask] = -remaining[valid_mask]
        return scores
    
    remaining = bins - item
    
    # Only bins with enough remaining capacity
    valid_mask = remaining >= -1e-12
    
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid_mask):
        return scores
    
    valid_remaining = remaining[valid_mask]
    n_valid = len(valid_remaining)
    
    if n_valid == 0:
        return scores
    
    # Base best-fit score: -remaining (higher = less remaining = preferred)
    base_score = -valid_remaining
    
    # Adaptive fragment penalty using smooth soft-min surrogate:
    # We want to avoid creating residuals that are too small to be useful.
    # Use a smooth penalty based on the distance to the "ideal" residual zone.
    
    # Threshold for "useful" residual: scale with item size
    # If item is large, a small absolute residual might still be too small
    # If item is small, even a moderate residual might be too small
    threshold = max(item * 0.2, 1.5)  # minimum useful residual
    
    # Smooth penalty: use a soft-min style approach
    # penalty = alpha * log(1 + exp(-beta * (residual - threshold))) 
    # This smoothly transitions from high penalty (small residual) to low penalty (large residual)
    # At residual = threshold, penalty = alpha * log(2)
    # As residual >> threshold, penalty -> 0
    # As residual << threshold, penalty -> alpha * beta * (threshold - residual)
    
    alpha = 2.0  # strength of fragment penalty
    beta = 5.0   # sharpness of transition
    
    residual = valid_remaining
    
    # Compute smooth penalty using log-sum-exp style formulation
    # x = beta * (residual - threshold)
    # penalty = alpha * log(1 + exp(-x)) = alpha * logaddexp(0, -x)
    x = beta * (residual - threshold)
    # Use logaddexp for numerical stability
    penalty = alpha * np.logaddexp(0, -x)
    
    # Combine: base score minus penalty
    score_valid = base_score - penalty
    
    scores[valid_mask] = score_valid
    
    return scores
# EVOLVE-BLOCK-END
