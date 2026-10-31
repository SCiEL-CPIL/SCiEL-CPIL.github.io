"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 140.0  # safety margin before 150s kill
_START_TIME = time.time()


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline guard: if we're too close to the limit, return a safe default
    if time.time() - _START_TIME > _DEADLINE:
        valid = bins >= item
        if not np.any(valid):
            return np.full_like(bins, -np.inf)
        scores = -(bins - item)
        scores[~valid] = -np.inf
        return scores

    # Determine which bins have enough remaining capacity
    valid = bins >= item
    
    if not np.any(valid):
        return np.full_like(bins, -np.inf)
        
    # Hybrid strategy:
    # - For small items: Best Fit (minimize leftover) to consolidate
    # - For large items: slightly prefer bins with more remaining space (Worst Fit tendency)
    #   to keep bins open for other large items and reduce fragmentation
    
    # Threshold: if item is > 30% of capacity (100), treat as "large"
    # Capacity is 100, so threshold at 30
    large_threshold = 30.0
    
    remaining = bins - item
    valid_remaining = remaining[valid]
    valid_bins = bins[valid]
    
    if item > large_threshold:
        # For large items: use a blend that slightly prefers bins with more remaining space
        # score = -remaining + beta * bins
        # This is like: score = -bins + item + beta * bins = item + (beta - 1) * bins
        # For beta = 0.3: score = item - 0.7 * bins
        # This still prefers tighter fits but less aggressively than pure best-fit
        beta = 0.3
        scores = item + (beta - 1.0) * valid_bins
    else:
        # For small items: pure Best Fit with slight superlinear penalty on large leftover
        # score = -remaining - alpha * remaining^2 / (remaining + delta)
        # This makes the penalty for large waste superlinear, reducing fragmentation
        delta = 1.0
        alpha = 0.01
        smooth_penalty = alpha * valid_remaining * valid_remaining / (valid_remaining + delta)
        scores = -valid_remaining - smooth_penalty
    
    # Mask invalid bins
    scores_full = np.full_like(bins, -np.inf)
    scores_full[valid] = scores
    
    return scores_full
# EVOLVE-BLOCK-END
