import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin for the given item.
    Higher score is better.
    
    Uses a fluid-flow-inspired heuristic: prioritize bins where the item
    constitutes a large fraction of the remaining capacity, minimizing
    relative waste. Combined with a small best-fit bias for tie-breaking.
    """
    n_bins = len(bins)
    scores = np.full(n_bins, -1e9, dtype=np.float64)
    
    # Find bins that can accommodate the item
    valid_mask = bins >= item
    
    if not np.any(valid_mask):
        return scores
        
    valid_remaining = bins[valid_mask]
    
    # Fluid flow score: item / remaining_capacity
    # Higher means the item fills a larger fraction of the bin
    # This minimizes the "relative waste" of opening/using a bin
    fluid_score = item / valid_remaining
    
    # Best fit component: minimize remaining after placement
    remaining_after = valid_remaining - item
    best_fit_score = -remaining_after / 100.0  # Scale to [-1, 0]
    
    # Combine: fluid score dominates, best-fit provides tie-breaking
    # The small weight on best-fit ensures it only matters when fluid scores are close
    combined = fluid_score + 0.01 * best_fit_score
    
    scores[valid_mask] = combined
    
    return scores
