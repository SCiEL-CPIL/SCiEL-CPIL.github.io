import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining capacity after placing the item
    # If bins[i] < item, the bin is invalid.
    # We want to minimize (bins[i] - item) for valid bins.
    # Score should be high for small (bins[i] - item).
    
    # Mask for valid bins
    valid = bins >= item
    
    # Calculate scores
    # For invalid bins, set score to -np.inf
    # For valid bins, score = -(bins - item)
    
    scores = np.full_like(bins, -np.inf, dtype=float)
    if np.any(valid):
        # Only compute for valid bins to avoid negative values affecting logic if we were doing something else
        # But -(bins - item) for valid bins is <= 0.
        # We want the highest score (closest to 0 or positive if we shifted).
        # Best-Fit: Minimize remaining space.
        # Score = item - bins[i]
        scores[valid] = item - bins[valid]
        
    return scores
