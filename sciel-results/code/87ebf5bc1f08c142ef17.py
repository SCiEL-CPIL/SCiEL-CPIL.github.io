"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Returns an array of scores; the item goes to the highest-scoring bin.
    
    Strategy: Best-Fit with a penalty for creating too-small residuals.
    We prefer bins where the remaining capacity after placing the item is
    as small as possible (best-fit), but we add a penalty if the residual
    is very small but positive (to avoid creating unusable fragments).
    """
    remaining = bins - item
    
    # Only bins with enough remaining capacity
    valid_mask = remaining >= -1e-12
    
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid_mask):
        return scores
    
    valid_remaining = remaining[valid_mask]
    n_valid = len(valid_remaining)
    
    if n_valid == 0:
        return scores
    
    # Best-fit: prefer smallest remaining capacity
    # Score = -remaining (higher is better, so smaller remaining = higher score)
    
    # Add a penalty for very small residuals (less than ~10% of item size)
    # This avoids creating fragments that can't hold any future item
    # The penalty is: -lambda * exp(-residual / threshold) for residual > 0
    # where threshold is proportional to item size
    
    threshold = max(item * 0.1, 0.5)  # minimum useful residual size
    
    # For bins with very small residual, apply a penalty
    # This encourages leaving either enough space or almost no space
    # Actually, for online bin packing, best-fit is generally strong.
    # Let's use a simple best-fit with a slight preference for not
    # creating extremely small residuals.
    
    # Score: -remaining - penalty
    # Penalty is high when residual is in (0, threshold)
    # Use a smooth penalty: penalty = alpha * (1 - residual/threshold)^2 for residual < threshold
    
    alpha = 2.0  # strength of fragment penalty
    
    residual = valid_remaining
    penalty = np.zeros(n_valid)
    
    # Apply penalty only for residuals in (0, threshold)
    small_mask = (residual > 0) & (residual < threshold)
    if np.any(small_mask):
        # Normalize residual to [0, 1] where 0 = threshold, 1 = 0
        norm_res = 1.0 - residual[small_mask] / threshold
        penalty[small_mask] = alpha * norm_res ** 2
    
    score_valid = -residual - penalty
    
    scores[valid_mask] = score_valid
    
    return scores

# EVOLVE-BLOCK-END
