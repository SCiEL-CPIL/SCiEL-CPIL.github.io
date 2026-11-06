import numpy as np
import time

# We assume capacity is 100 as per task description
CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining space
    remaining = bins - item
    
    # Initialize scores with -inf for invalid bins
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # Find valid indices (bins with enough remaining capacity)
    valid_mask = remaining >= 0
    valid_idx = np.where(valid_mask)[0]
    
    if len(valid_idx) == 0:
        return scores

    valid_remaining = remaining[valid_idx]
    
    # Hybrid strategy based on item size relative to capacity
    # For large items (> 0.7 * CAPACITY), use Worst-Fit (maximize remaining)
    # For small items (< 0.3 * CAPACITY), use Best-Fit (minimize remaining)
    # For medium items, use a balanced approach
    
    if item > 0.7 * CAPACITY:
        # Large item: Worst-Fit - prefer bins with most remaining space
        # This avoids creating new bins unnecessarily
        scores[valid_idx] = valid_remaining
    elif item < 0.3 * CAPACITY:
        # Small item: Best-Fit - prefer bins with least remaining space
        # This packs small items tightly
        scores[valid_idx] = -valid_remaining
    else:
        # Medium item: Use a smooth hybrid
        # Balance between best-fit and worst-fit
        # Score based on proximity to half the remaining capacity
        # Prefer bins where remaining is close to item size (tight fit)
        # or use a combination
        
        # Use best-fit for medium items as well, but with a slight preference
        # for bins that are not too empty
        # Score = -remaining + small bonus for larger bins
        # This creates a smooth transition
        
        # Normalize remaining to [0, 1] range for the valid bins
        min_rem = np.min(valid_remaining)
        max_rem = np.max(valid_remaining)
        range_rem = max_rem - min_rem if max_rem > min_rem else 1.0
        
        normalized = (valid_remaining - min_rem) / range_rem
        
        # Best-fit dominates, but with a small weight for worst-fit
        # to avoid over-packing into already full bins
        scores[valid_idx] = -valid_remaining + 0.1 * valid_remaining
        
    return scores
