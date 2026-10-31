# EVOLVE-BLOCK-START
import numpy as np

# Global state for adaptive estimation of item distribution
_item_count = 0
_item_sum = 0.0
_item_sum_sq = 0.0
_default_mean = 15.0  # Fallback estimate for Weibull mean if no data

def _update_stats(item: float):
    global _item_count, _item_sum, _item_sum_sq
    _item_count += 1
    _item_sum += item
    _item_sum_sq += item * item

def _get_mean() -> float:
    global _item_count, _item_sum
    if _item_count == 0:
        return _default_mean
    return _item_sum / _item_count

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin based on the negative squared deviation of the new remaining space
    from the estimated mean item size. This prioritizes bins that leave a remaining
    space close to the expected size of the next item, maximizing the probability
    of accepting subsequent items while minimizing wasted space.
    """
    _update_stats(item)
    
    mean_item = _get_mean()
    
    # Initialize scores to -infinity
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    # Find valid bins (enough capacity for current item)
    valid_mask = bins >= item
    
    if not np.any(valid_mask):
        return scores
    
    # Calculate remaining space after placing the current item
    new_remaining = bins[valid_mask] - item
    
    # Calculate deviation from the target remaining space (mean item size)
    diff = new_remaining - mean_item
    
    # Score is negative squared deviation
    # Higher score (closer to 0) means new_remaining is closer to mean_item
    scores[valid_mask] = -(diff * diff)
    
    return scores
# EVOLVE-BLOCK-END
