import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin for the given item.
    Higher score is better.
    Uses a hybrid strategy with a global variance-based adjustment:
    - For large items (> 50% of capacity), use Worst Fit (maximize remaining space).
    - For small items (<= 50% of capacity), use Best Fit (minimize remaining space).
    - Both are adjusted by a term that favors bins whose remaining capacity is 
      close to the mean of all open bins, reducing fragmentation.
    """
    n_bins = len(bins)
    # Initialize all scores to a very low value (invalid for selection)
    scores = np.full(n_bins, -1e9, dtype=np.float64)
    
    # Find bins that can accommodate the item
    valid_mask = bins >= item
    
    # If no valid bins, return the low scores (selection will fail or pick arbitrary, 
    # but evaluator should handle this or input is valid)
    if not np.any(valid_mask):
        return scores
        
    # Extract remaining capacities of valid bins
    valid_remaining = bins[valid_mask]
    
    # Capacity is 100. Threshold is 50.
    threshold = 50.0
    
    # Compute global statistics for variance-based adjustment
    if n_bins > 1:
        mean_remaining = np.mean(bins)
        # We want to favor bins that are "typical" (close to the mean) to reduce fragmentation
        # This is a smooth surrogate for minimizing variance
        variance_term = -np.abs(bins - mean_remaining)
    else:
        variance_term = np.zeros(n_bins)
    
    # Weight for the variance adjustment term
    # Small items benefit more from consolidation (Best Fit), so we can weight the variance term differently
    # Large items benefit from keeping bins open (Worst Fit), so the variance term should be less dominant
    
    if item > threshold:
        # Large item: Worst Fit + mild variance adjustment
        # We want the bin with the MOST remaining space, but also prefer bins that are not extreme outliers
        # Use a smaller weight for the variance term to not override the Worst Fit logic
        scores[valid_mask] = valid_remaining + 0.1 * variance_term[valid_mask]
    else:
        # Small item: Best Fit + stronger variance adjustment
        # We want the bin with the LEAST remaining space, but also prefer bins that are not extreme outliers
        # Use a larger weight for the variance term to encourage consolidation
        scores[valid_mask] = -valid_remaining + 0.5 * variance_term[valid_mask]
        
    return scores
