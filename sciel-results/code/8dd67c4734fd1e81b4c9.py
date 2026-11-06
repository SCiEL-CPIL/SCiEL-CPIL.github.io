"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state for adaptive parameters
_start_time = time.time()
_deadline = 145.0  # seconds (safety margin before 150s kill)

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score = better choice.
    
    Uses an Aggressive Best-Fit heuristic.
    The score is the negative of the relative gap raised to a power > 1.
    This sharpens the preference for bins where the item fits very tightly,
    which is generally optimal for minimizing the number of bins in 
    right-skewed distributions like Weibull.
    
    The previous version used a Gaussian penalty for "danger zone" gaps,
    which may have introduced noise. This version relies on a smooth,
    monotonic penalty for larger gaps.
    """
    # Time guard (though this operation is O(N) and very fast)
    if time.time() > _deadline:
        # Fallback to simple best-fit if we are running out of time
        remaining = bins - item
        valid = remaining >= 0
        score = np.where(valid, -remaining, -np.inf)
        return score

    remaining = bins - item
    valid = remaining >= 0
    
    if not np.any(valid):
        return np.full(len(bins), -np.inf)
    
    # Calculate relative gap: how much extra space is left relative to item size
    # Scale by item to make scores comparable
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_gap = np.where(valid, remaining / max(item, 1e-10), 0.0)
    
    # Aggressive Best-Fit Score
    # Power > 1 penalizes larger gaps more heavily than linear Best-Fit
    # This helps avoid "fragmentation" where many bins have moderate gaps
    alpha = 1.2
    score = -np.power(rel_gap, alpha)
    
    # Ensure invalid bins have -inf score
    score = np.where(valid, score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
