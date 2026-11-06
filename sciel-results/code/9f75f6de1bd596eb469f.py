"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the item.
    Higher score = preferred bin.
    
    Strategy: Harmonic-mean based scoring with smooth best-fit regularization.
    
    The harmonic mean of remaining capacity (rem) and item size naturally
    penalizes both tiny gaps (fragmentation) and large remaining spaces
    (waste). We add a smooth best-fit term to break ties and encourage
    tighter packing.
    
    Score = -harmonic_mean(rem, item) - alpha * sqrt(rem)
    
    Where:
    - harmonic_mean = 2 * rem * item / (rem + item)
    - alpha = 0.1 (small weight for best-fit preference)
    - This is smooth and differentiable everywhere for rem > 0
    """
    t0 = time.time()
    
    # Bins that can fit the item
    remaining = bins - item
    valid = remaining >= 0
    
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    if np.any(valid):
        rem = remaining[valid]
        eps = 1e-9
        
        # Harmonic mean of remaining capacity and item size
        # Maximized when rem ≈ item, penalizes both extremes
        harmonic_mean = 2.0 * rem * item / (rem + item + eps)
        
        # Smooth best-fit term: prefer smaller remainders
        alpha = 0.1
        best_fit_term = -alpha * np.sqrt(rem + eps)
        
        scores[valid] = -harmonic_mean + best_fit_term
    
    # Deadline guard
    if time.time() - t0 > 110:
        # Fallback: simple best-fit
        scores = np.where(valid, -remaining, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
