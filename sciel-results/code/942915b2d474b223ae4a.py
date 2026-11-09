# EVOLVE-BLOCK-START
import numpy as np
import time

# Weibull parameters (typical for bin packing benchmarks)
# Mean of Weibull(k, scale) = scale * gamma(1 + 1/k)
# For k=2.0, scale=15.0: mean ≈ 15.0 * gamma(1.5) ≈ 15.0 * 0.886 ≈ 13.3
# This gives a reasonable distribution of item sizes around 13-14
_WEIBULL_K = 2.0
_WEIBULL_SCALE = 15.0
_EXPECTED_NEXT = _WEIBULL_SCALE * 0.886  # approx gamma(1 + 1/2)

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the item.
    Higher score = preferred bin.
    
    Strategy: Next-Item Compatibility.
    Score each bin by how well its remaining capacity (after placing current item)
    aligns with the expected size of the next item. This prioritizes bins that
    will remain useful for future items rather than just minimizing current waste.
    """
    remaining = bins - item
    valid = remaining >= 0
    
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    if np.any(valid):
        rem = remaining[valid]
        
        # The remaining capacity after placing the current item
        # We want this to be close to the expected next item size
        # If rem ≈ expected_next, the bin is well-positioned for the next item
        
        # Score: negative absolute difference between remaining and expected next
        # This favors bins where remaining ≈ expected_next
        # Add a small bonus for larger remaining (more flexibility)
        
        # Primary: minimize |rem - expected_next|
        # Secondary: slight preference for rem > expected_next (more room)
        diff = np.abs(rem - _EXPECTED_NEXT)
        
        # Bonus: if rem is significantly larger than expected_next, 
        # the bin has more flexibility (can fit larger items too)
        # But if rem is much smaller, it may become dead space
        flexibility_bonus = np.maximum(0, rem - _EXPECTED_NEXT) * 0.1
        
        # Combine: primary penalty for mismatch, secondary flexibility bonus
        scores[valid] = -diff + flexibility_bonus
    
    return scores
# EVOLVE-BLOCK-END
