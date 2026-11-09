"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Worst-Fit with a smooth nonlinear transform.
    We want to prefer bins with MORE remaining capacity (worst-fit),
    but use a power-law transform to capture diminishing marginal value.
    
    For worst-fit, we want higher remaining capacity -> higher score.
    The remaining capacity is (bins - item).
    
    Using a power-law: score = (bins - item)^p for p > 0
    This gives higher scores to bins with more remaining space,
    but with diminishing returns as the space gets very large.
    
    However, we need to be careful: if p is small, the differences
    between bins are compressed. If p is large, they are amplified.
    
    A good choice might be p=0.5 (square root) which gives a smooth
    concave transform, or even p=1.0 which is just linear worst-fit.
    
    Let's use p=0.5 for a smooth nonlinear transform.
    
    For bins that don't have enough capacity, we return -inf so they
    are never selected.
    """
    remaining = bins - item
    # Mask out bins that can't fit the item
    valid = remaining >= 0
    scores = np.full_like(remaining, -np.inf, dtype=np.float64)
    if np.any(valid):
        # Use square root of remaining capacity as score
        # This gives higher score to bins with more remaining space
        scores[valid] = np.sqrt(remaining[valid])
    return scores
# EVOLVE-BLOCK-END
