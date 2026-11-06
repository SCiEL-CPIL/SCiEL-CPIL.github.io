"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Score each bin with enough remaining capacity.
    
    Uses a best-fit strategy with a slight preference for bins that leave
    a moderate gap (not too tight, not too loose) to balance density
    with future flexibility.
    """
    remaining = bins - item
    mask = remaining >= 0
    
    scores = np.full(bins.shape, -np.inf)
    if not np.any(mask):
        return scores
    
    r = remaining[mask]
    
    # Best-fit: prefer smallest remaining
    # Add a small term that penalizes extremely tight fits (remaining << item)
    # to preserve some flexibility for future items
    # Use: score = -r + small_correction
    # The correction is zero when r is "reasonable" and negative when r is tiny
    
    # Normalize by item to make scale-invariant
    ratio = r / (item + 1e-12)
    
    # Primary: minimize remaining (best-fit)
    # Secondary: if remaining is very small (< 0.1 * item), add small penalty
    # to avoid creating unusable gaps
    penalty = np.where(ratio < 0.05, (0.05 - ratio) * item * 0.1, 0.0)
    
    scores[mask] = -r - penalty
    
    return scores
# EVOLVE-BLOCK-END
