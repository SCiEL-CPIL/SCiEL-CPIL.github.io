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
    
    Strategy: Sharpened Best-Fit.
    We score bins based on the ratio of their remaining capacity to the maximum
    remaining capacity among valid bins. We apply a quadratic penalty to this ratio.
    This strongly favors the bin with the least remaining capacity (Best-Fit),
    while making the preference for the tightest fit significantly stronger than
    linear Best-Fit. This helps in consolidating items and reducing the total
    number of bins used, which directly minimizes the excess over the L1 lower bound.
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
    
    # Find the maximum remaining capacity among valid bins
    max_rf = np.max(valid_remaining)
    
    # If max_rf is very small, all valid bins are effectively "full"
    # We can just pick any, or the one with smallest remaining (Best Fit)
    if max_rf < 1e-9:
        # All remaining are ~0. Score them all equally high, or use -remaining for tie break
        scores[valid_mask] = -valid_remaining
        return scores
    
    # Normalize remaining capacity to [0, 1] range
    # 0 means "tightest fit" (smallest remaining), 1 means "loosest fit" (largest remaining)
    # Note: min_remaining might not be 0.
    # We want the tightest to have score close to 0 (or high positive if we invert).
    # Let's use: score = - (normalized_remaining)^p
    # where normalized_remaining = remaining / max_rf
    # This maps [min_rf, max_rf] to [min_rf/max_rf, 1.0]
    # The tightest (min_rf) will have the smallest normalized value, thus highest score (least negative).
    
    normalized_remaining = valid_remaining / max_rf
    
    # Apply quadratic penalty (p=2.0) to sharpen the preference for tight fits
    # This makes the difference between a "good" fit and a "slightly worse" fit more significant
    score_valid = - (normalized_remaining ** 2)
    
    scores[valid_mask] = score_valid
    
    return scores

# EVOLVE-BLOCK-END
