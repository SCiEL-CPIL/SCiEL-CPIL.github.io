import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining capacity for each bin
    remaining = bins - item
    
    # Initialize scores with -infinity for bins that cannot hold the item
    scores = np.full_like(remaining, -np.inf, dtype=float)
    
    # Mask for bins that have enough capacity
    valid_mask = remaining >= 0
    
    if not np.any(valid_mask):
        return scores
    
    # For valid bins, calculate the adaptive weight
    # We want Best Fit (w=1) for large items and Neutral (w=0) for small items.
    # Sigmoid transition centered around a threshold.
    # Capacity is 100. 
    # Let's set the transition center at 15.0 and steepness k=0.4
    # w = 1 / (1 + exp(-k * (item - center)))
    
    center = 15.0
    k = 0.4
    weight = 1.0 / (1.0 + np.exp(-k * (item - center)))
    
    # Apply the weight to the negative remaining capacity.
    # If weight is 1, score = -remaining (Best Fit).
    # If weight is 0, score = 0 (Neutral, ties broken by index -> First Fit).
    scores[valid_mask] = -weight * remaining[valid_mask]
    
    return scores
