import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    mask = remaining >= 0
    scores = np.full_like(bins, -np.inf)
    
    # Adaptive strategy:
    # For larger items, use Worst-Fit to preserve large contiguous spaces.
    # For smaller items, use Best-Fit to pack tightly and utilize small gaps.
    # Threshold set to 50% of capacity (50).
    
    if item > 50:
        scores[mask] = remaining[mask]
    else:
        scores[mask] = -remaining[mask]
        
    return scores
