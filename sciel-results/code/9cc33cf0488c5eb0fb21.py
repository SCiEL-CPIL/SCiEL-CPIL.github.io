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
    
    Strategy: Best Fit with a small penalty for creating bins with very little
    remaining space (dust penalty). This avoids wasting bins on tiny remnants.
    """
    if time.time() - _START_TIME > _MAX_EVAL_TIME:
        remaining = bins - item
        scores = np.where(remaining >= 0, -remaining, -np.inf)
        return scores

    remaining_after = bins - item
    feasible = remaining_after >= 0

    if not np.any(feasible):
        return np.full(len(bins), -np.inf)

    # Base score: Best Fit (minimize remaining space after placement)
    scores = np.full(len(bins), -np.inf)
    
    # Dust threshold: bins with < 2% of capacity remaining after placement
    # get a small penalty to avoid creating unusable remnants
    dust_threshold = 2.0  # 2% of capacity 100
    penalty_weight = 0.5
    
    rem = remaining_after[feasible]
    base_scores = -rem
    dust_penalty = penalty_weight * np.maximum(0.0, dust_threshold - rem)
    scores[feasible] = base_scores + dust_penalty
    
    return scores
# EVOLVE-BLOCK-END
