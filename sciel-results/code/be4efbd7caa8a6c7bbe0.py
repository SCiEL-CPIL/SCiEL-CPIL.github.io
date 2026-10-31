import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid_mask = remaining >= 0
    
    if not np.any(valid_mask):
        return np.full_like(bins, -np.inf)
    
    scores = np.full_like(bins, -np.inf)
    
    # Smooth transition between Best Fit and Worst Fit
    # For small items, prefer Best Fit (minimize remaining)
    # For large items, prefer Worst Fit (maximize remaining)
    # Use a smooth sigmoid-like blend around item=30
    
    capacity = 100.0
    # Blend factor: 0 = pure Best Fit, 1 = pure Worst Fit
    # Smooth transition centered at 30, width ~10
    t = 1.0 / (1.0 + np.exp(-(item - 30.0) / 10.0))
    
    # Best Fit score: -remaining (lower remaining is better)
    # Worst Fit score: +remaining (higher remaining is better)
    # Blend: t * worst_fit_score + (1-t) * best_fit_score
    best_fit_score = -remaining
    worst_fit_score = remaining
    
    blended_score = t * worst_fit_score + (1.0 - t) * best_fit_score
    
    scores[valid_mask] = blended_score[valid_mask]
    
    return scores
