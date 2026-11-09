import numpy as np
import time

# Module-level state for deadline tracking
_start_time = time.time()

# Estimated median item size for Weibull sequences with capacity 100
# Weibull with shape ~2, scale ~5 gives median around 3-5
_MEDIAN_ITEM_SIZE = 4.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Returns an array of one score per bin (higher = better).
    
    Strategy: Adaptive Best-Fit with gap-awareness for Weibull-distributed items.
    - Prefer bins where item fits with minimal leftover (best-fit).
    - Penalize creating tiny gaps that won't fit future Weibull items.
    - For very small items, prefer fuller bins to reduce fragmentation.
    """
    # Deadline guard: if we're running long, fall back to simple best-fit
    elapsed = time.time() - _start_time
    if elapsed > 145.0:
        # Simple best-fit as fallback
        remaining = bins - item
        mask = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        scores[mask] = -remaining[mask]
        return scores

    remaining = bins - item
    mask = remaining >= 0
    
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(mask):
        return scores
    
    rem = remaining[mask]
    
    # "Next-Item Fit" proxy: score based on how well the remaining capacity
    # matches the median expected future item size
    # We want rem to be close to _MEDIAN_ITEM_SIZE so that a typical next item
    # fits with minimal waste. Use negative squared difference as score.
    
    # Smooth nonlinear scoring: prefer rem values near the median item size
    # This avoids fragmentation (too small gaps) and underutilization (too large gaps)
    diff = rem - _MEDIAN_ITEM_SIZE
    next_item_fit_score = -diff * diff
    
    # Additional consideration: if rem is very small (< 1), it's essentially wasted
    # Add a smooth penalty for very small remaining capacities
    small_rem_penalty = np.where(rem < 1.0, rem * 2.0, 0.0)
    
    # For large items, we should still prefer best-fit behavior
    # Blend the next-item-fit score with best-fit based on item size
    if item > 8.0:
        # Large items: primarily best-fit
        best_fit_score = -rem
        final_scores = 0.8 * best_fit_score + 0.2 * next_item_fit_score - small_rem_penalty
    elif item > 4.0:
        # Medium items: balance between best-fit and next-item-fit
        best_fit_score = -rem
        final_scores = 0.5 * best_fit_score + 0.5 * next_item_fit_score - small_rem_penalty
    else:
        # Small items: primarily next-item-fit to reduce fragmentation
        final_scores = next_item_fit_score - small_rem_penalty
    
    scores[mask] = final_scores
    
    return scores
