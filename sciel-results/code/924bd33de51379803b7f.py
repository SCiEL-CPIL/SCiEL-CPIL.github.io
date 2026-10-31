import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid_mask = remaining >= 0
    
    if not np.any(valid_mask):
        return np.full_like(bins, -np.inf)
    
    scores = np.full_like(bins, -np.inf)
    
    # Adjusted thresholds: Lean towards Best Fit (small w) for more of the range.
    # Best Fit preserves larger holes, which is generally better for packing.
    # Worst Fit is only for very large items where we might want to "use up" a large hole 
    # to prevent it from being wasted, or to consolidate large items? 
    # Actually, let's test if pure Best Fit is better.
    # But the directive says "Keep this design". So we keep the hybrid.
    
    low = 40.0
    high = 85.0
    
    if item <= low:
        w = 0.0
    elif item >= high:
        w = 1.0
    else:
        t = (item - low) / (high - low)
        # Sharper transition
        w = t * t * (3.0 - 2.0 * t)
        
    # Blend
    scores[valid_mask] = (2.0 * w - 1.0) * remaining[valid_mask]
    
    return scores
