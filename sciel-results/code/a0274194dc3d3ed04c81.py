import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    if time.time() - _START_TIME > _DEADLINE:
        # Fallback to simple Best Fit
        remaining = bins - item
        valid = remaining >= 0
        scores = np.full_like(bins, -np.inf)
        if np.any(valid):
            scores[valid] = 1.0 / (remaining[valid] + 1e-9)
        return scores

    remaining = bins - item
    valid = remaining >= 0
    
    if not np.any(valid):
        return np.full_like(bins, -np.inf)
        
    scores = np.full_like(bins, -np.inf)
    
    # Target Fit Heuristic
    # Prefer gaps close to T.
    # For g < T, score increases with g (Worst Fit to avoid dead space).
    # For g >= T, score decreases with g (Best Fit to consolidate).
    # Continuity at T.
    T = 10.0
    g = remaining[valid]
    
    # Calculate scores for valid bins
    # Use vectorized operations
    scores_valid = np.empty_like(g)
    
    mask_small = g < T
    mask_large = ~mask_small
    
    # Small gaps: score = g (linear increase)
    scores_valid[mask_small] = g[mask_small]
    
    # Large gaps: score = T^2 / g (hyperbolic decrease)
    # Avoid division by zero (though g >= T > 0 here)
    scores_valid[mask_large] = (T * T) / g[mask_large]
    
    scores[valid] = scores_valid
    
    return scores
