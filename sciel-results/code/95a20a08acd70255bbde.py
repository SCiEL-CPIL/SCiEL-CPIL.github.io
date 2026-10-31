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
    
    Strategy: Best-Fit with consolidation bias.
    Place the item in the bin with the smallest remaining capacity that can fit it,
    with a small bonus for bins that are already partially filled to encourage
    consolidation and reduce the number of bins used.
    """
    if time.time() - _START_TIME > _MAX_EVAL_TIME:
        remaining = bins - item
        scores = np.where(remaining >= 0, -remaining, -np.inf)
        return scores

    remaining_after = bins - item
    feasible = remaining_after >= 0

    if not np.any(feasible):
        return np.full(len(bins), -np.inf)

    # Best-Fit: prefer bins with less remaining capacity (higher score = less remaining)
    # Add a small consolidation bias: slightly prefer bins that are already in use
    # (i.e., bins with some items already, which have less remaining capacity)
    scores = np.full(len(bins), -np.inf)
    
    # Primary: minimize remaining capacity after placement
    scores[feasible] = -remaining_after[feasible]
    
    # Secondary: small bonus for bins that are more "full" (less remaining)
    # This encourages packing items together to free up entire bins
    # Use a small epsilon to not overwhelm the primary criterion
    epsilon = 0.01
    scores[feasible] += epsilon * (100.0 - remaining_after[feasible])
    
    return scores
# EVOLVE-BLOCK-END
