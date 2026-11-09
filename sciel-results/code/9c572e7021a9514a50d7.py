"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.

    Strategy: Worst-Fit.
    We prioritize bins with the largest remaining capacity.
    This helps to prevent fragmentation by keeping bins more uniformly filled,
    which is often beneficial for distributions with many small items (Weibull).
    """
    remaining = bins - item
    valid = remaining >= 0
    
    score = np.full_like(bins, -np.inf, dtype=float)
    
    if not np.any(valid):
        return score

    # Worst-Fit: Higher remaining capacity is better.
    # We assign the remaining capacity as the score.
    score[valid] = remaining[valid]
    
    return score
# EVOLVE-BLOCK-END
