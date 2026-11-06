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
    
    Strategy: Best-Fit with adaptive fragment penalty and smooth quadratic regularization.
    - Base score: -remaining (prefer bins with least remaining capacity)
    - Penalty: smooth penalty for creating very small residuals (fragments)
      that are unlikely to be useful for future items
    - Quadratic regularization: gently prefer more uniform filling to avoid
      creating a few very full bins and many empty ones
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
    
    # Adaptive fragment penalty:
    # We want to avoid creating residuals that are too small to be useful
    # but not zero. A residual of ~0 is fine (bin is full), but a residual
    # of 1-2 units when items are typically 10-50 is wasteful.
    
    # Threshold for "useful" residual: scale with item size
    threshold = max(item * 0.15, 1.0)  # minimum useful residual
    
    # Penalty: smooth quadratic penalty for residuals in (0, threshold)
    # penalty = alpha * (1 - residual/threshold)^2 for 0 < residual < threshold
    # This is 0 at residual=threshold and alpha at residual=0
    
    alpha = 1.5  # strength of fragment penalty
    
    residual = valid_remaining
    penalty = np.zeros(n_valid)
    
    # Apply penalty only for residuals in (0, threshold)
    small_mask = (residual > 1e-9) & (residual < threshold)
    if np.any(small_mask):
        norm_res = 1.0 - residual[small_mask] / threshold
        penalty[small_mask] = alpha * norm_res ** 2
    
    # Quadratic regularization term:
    # Gently discourage bins with very large remaining capacity relative to
    # the maximum, promoting more uniform bin utilization.
    # This is a soft preference, not a hard constraint.
    max_rf = np.max(valid_remaining)
    if max_rf > 1e-12:
        quad_term = (valid_remaining / max_rf) ** 2
    else:
        quad_term = np.zeros(n_valid)
    
    gamma = 0.2  # weight for quadratic regularization (small, to not override best-fit)
    
    # Combine: base score minus penalty minus quadratic regularization
    score_valid = base_score - penalty - gamma * quad_term
    
    scores[valid_mask] = score_valid
    
    return scores
# EVOLVE-BLOCK-END
