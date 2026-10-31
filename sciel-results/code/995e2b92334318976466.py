"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the arriving item.
    
    Strategy: Smooth Target-Fit with Dead-Space Avoidance.
    - Uses a target remainder R_target that depends on the item size.
    - For small items, R_target is small (e.g., 5) to pack tightly but avoid unusable gaps (<5).
    - For large items, R_target is large (e.g., 100) to preserve flexibility (Worst-Fit).
    - Score is - (Remaining - R_target)^2, which is a smooth surrogate for minimizing
      deviation from the ideal remainder.
    """
    start_time = time.time()
    deadline = 115.0  # 120s limit, 5s safety margin
    
    capacity = 100.0
    
    # Compute remaining capacity if we place the item in each bin
    remaining = bins - item
    
    # Valid bins are those with remaining >= 0
    valid_mask = remaining >= 0
    
    # If no valid bin, return -inf for all
    if not np.any(valid_mask):
        return np.full_like(bins, -np.inf)
    
    # Initialize scores to -inf
    scores = np.full_like(bins, -np.inf)
    
    # Item size ratio
    item_ratio = item / capacity  # 0 to 1
    
    # Smooth transition for target remainder
    # Small items -> R_target ~ 5.0 (avoid dead space, pack tight)
    # Large items -> R_target ~ 100.0 (worst fit, preserve space)
    
    transition_center = 0.5
    transition_width = 0.1
    
    # Sigmoid for transition
    x = (item_ratio - transition_center) / transition_width
    # Clip to avoid overflow in exp
    x = np.clip(x, -10.0, 10.0)
    t = 1.0 / (1.0 + np.exp(-x))
    
    R_min = 5.0
    R_max = 100.0
    
    R_target = R_min + (R_max - R_min) * t
    
    # Score: - (Remaining - R_target)^2
    # We only compute for valid bins
    r_valid = remaining[valid_mask]
    scores[valid_mask] = - (r_valid - R_target) ** 2
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores

# EVOLVE-BLOCK-END
