"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin with enough remaining capacity.
    Uses a Quadratic Best Fit heuristic: prefers bins that leave the least
    remaining capacity, with a quadratic penalty for larger remainders.
    """
    # Check deadline
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: return Best Fit scores immediately
        remaining = bins - item
        return np.where(remaining >= 0, -remaining, -np.inf)

    remaining = bins - item
    # Mask bins that can't fit the item
    valid_mask = remaining >= 0
    
    # Calculate scores
    # Quadratic Best Fit: score = -(remaining)**2
    # This strongly prefers the tightest fit.
    scores = np.where(valid_mask, -(remaining ** 2), -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
