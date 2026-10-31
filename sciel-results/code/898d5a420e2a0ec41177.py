"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Pure Best-Fit.
    Place the item in the bin with the smallest remaining capacity that can fit it.
    This minimizes the waste in the chosen bin, which generally leads to better
    consolidation and fewer total bins used compared to First-Fit or Worst-Fit.
    """
    if time.time() - _START_TIME > _MAX_EVAL_TIME:
        remaining = bins - item
        scores = np.where(remaining >= 0, -remaining, -np.inf)
        return scores

    remaining_after = bins - item
    feasible = remaining_after >= 0

    if not np.any(feasible):
        return np.full(len(bins), -np.inf)

    scores = np.full(len(bins), -np.inf)
    scores[feasible] = -remaining_after[feasible]
    
    return scores
# EVOLVE-BLOCK-END
