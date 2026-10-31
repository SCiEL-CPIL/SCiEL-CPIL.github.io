"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the arriving item.
    
    Strategy: Hybrid Best/Worst Fit with smooth transition based on item size.
    - Large items (> 50% of capacity): prefer Worst-Fit (maximize remaining space)
      to preserve flexibility for future items.
    - Small items (<= 50% of capacity): prefer Best-Fit (minimize remaining space)
      to pack tightly and reduce fragmentation.
    
    The transition is smooth to avoid discontinuities.
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
    
    # Initialize scores
    scores = np.full_like(bins, -np.inf)
    
    # Item size ratio: how large is the item relative to capacity
    item_ratio = item / capacity  # 0 to 1
    
    # Smooth transition: weight for best-fit vs worst-fit
    # For small items (item_ratio < 0.5), prefer best-fit (minimize remaining)
    # For large items (item_ratio > 0.5), prefer worst-fit (maximize remaining)
    # Use a smooth sigmoid-like transition around 0.5
    
    # best_fit_weight: 1.0 for small items, 0.0 for large items
    # worst_fit_weight: 0.0 for small items, 1.0 for large items
    # Transition centered at 0.5, width ~0.2
    
    transition_center = 0.5
    transition_width = 0.2
    
    # Smooth step function
    x = (item_ratio - transition_center) / transition_width
    # Clamp to avoid extreme values
    x = np.clip(x, -5.0, 5.0)
    # Sigmoid: 0 for x << 0, 1 for x >> 0
    worst_fit_weight = 1.0 / (1.0 + np.exp(-x))
    best_fit_weight = 1.0 - worst_fit_weight
    
    # For valid bins, compute score
    # Best-fit: minimize remaining -> score = -remaining
    # Worst-fit: maximize remaining -> score = +remaining
    
    # Combined score:
    # score = best_fit_weight * (-remaining) + worst_fit_weight * (remaining)
    #       = remaining * (worst_fit_weight - best_fit_weight)
    #       = remaining * (2 * worst_fit_weight - 1)
    
    weight = 2.0 * worst_fit_weight - 1.0  # -1 for pure best-fit, +1 for pure worst-fit
    
    scores[valid_mask] = weight * remaining[valid_mask]
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores

# EVOLVE-BLOCK-END
