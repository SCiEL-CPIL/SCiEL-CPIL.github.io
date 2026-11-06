"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for online bin packing.
    
    The evaluator picks the bin with the highest score among bins with enough remaining capacity.
    We want to minimize the number of bins used (excess over L1 lower bound).
    
    Strategy: Enhanced Best-Fit using a logarithmic penalty on remaining space.
    The log1p function provides a strong preference for bins with very small remaining space,
    effectively acting as a smooth "tightest fit" selector. This helps consolidate items
    and reduce the number of partially filled bins, which is crucial for minimizing
    the total number of bins used.
    
    A small bonus is added to bins that are nearly full (remaining < 5%) to encourage
    closing bins.
    """
    # Deadline guard
    start_time = time.time()
    
    # Calculate remaining capacity for each bin
    remaining = bins - item
    
    # Filter: only consider bins with enough remaining capacity
    valid_mask = remaining >= 0
    
    # Initialize scores with -inf for invalid bins
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid_mask):
        return scores
    
    valid_remaining = remaining[valid_mask]
    
    # Use log1p for a smooth, convex penalty on remaining space.
    # -log1p(remaining) is high for small remaining, low for large remaining.
    # This strongly favors the bin with the least remaining space (Best Fit).
    base_score = -np.log1p(valid_remaining)
    
    # Bonus for bins that are almost full (remaining < 5.0, i.e., 5% of capacity 100)
    # This encourages completing bins to reduce the count of open bins.
    threshold = 5.0
    bonus = 0.1 * (valid_remaining < threshold).astype(float)
    
    final_scores = base_score + bonus
    
    scores[valid_mask] = final_scores
    
    # Safety check for time
    if time.time() - start_time > 1.0:
        pass # Return whatever we have, which is already computed
        
    return scores
# EVOLVE-BLOCK-END
