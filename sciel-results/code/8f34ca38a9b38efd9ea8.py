import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin before 150s limit

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Structural change: Use a 'gap quality' scoring that considers how the
    remaining gap after placing the item compares to the distribution of
    remaining gaps across all valid bins. This is a different modelling family
    from pure Best/Worst Fit — it uses relative gap positioning rather than
    absolute gap size.
    
    Idea: For Weibull-distributed items (skewed, many small), gaps that are
    'typical' of the current gap distribution are more likely to be filled
    by future items. Gaps that are outliers (too small or too large) are
    less useful. Score = -|log(remaining) - log(median_remaining)|, which
    rewards gaps close to the median gap size.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        remaining = bins - item
        scores = np.full_like(bins, -1e9, dtype=float)
        valid = remaining >= 0
        # Fallback: simple best fit
        scores[valid] = -remaining[valid]
        return scores

    remaining = bins - item
    valid = remaining >= 0
    scores = np.full_like(bins, -1e9, dtype=float)
    
    if not np.any(valid):
        return scores

    valid_remaining = remaining[valid]
    valid_indices = np.where(valid)[0]

    # Structural change: score based on proximity to the median gap size
    # This is a different family: it's a "typicality" score rather than
    # a "minimize/maximize gap" score.
    
    # Use median (robust to outliers) as the target gap size
    median_rem = np.median(valid_remaining)
    
    # Avoid log(0)
    safe_rem = np.maximum(valid_remaining, 1e-9)
    safe_median = max(median_rem, 1e-9)
    
    # Score: negative absolute log-distance to median
    # This rewards gaps that are "typical" and penalizes outliers
    log_dist = np.abs(np.log(safe_rem) - np.log(safe_median))
    
    # Add a small bonus for larger gaps (to avoid all bins having tiny gaps)
    # But the main signal is typicality
    scores[valid_indices] = -log_dist + 0.01 * valid_remaining
    
    return scores
