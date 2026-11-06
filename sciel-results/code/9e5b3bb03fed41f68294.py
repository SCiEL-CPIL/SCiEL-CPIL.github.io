"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state for adaptive parameters
_start_time = time.time()
_deadline = 150.0  # seconds

# We'll use a refined heuristic that combines relative best-fit with a smooth
# penalty for creating gaps that are likely unusable for future Weibull items.
# The key insight: for Weibull distributions, small items are more common, so
# we should prefer placements that leave gaps that can still accommodate small items.

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score = better choice.
    
    Uses a combination of:
    1. Relative Best-Fit: minimize remaining/item (primary)
    2. Smooth gap penalty: penalize remaining space that's too large relative
       to typical future item sizes, using a smooth surrogate for the min operation
    
    The smooth penalty uses a log-sum-exp inspired approach to create a
    differentiable approximation that penalizes gaps likely to be wasted.
    """
    elapsed = time.time() - _start_time
    if elapsed > _deadline - 0.5:
        # Safety margin: return a simple best-fit to avoid timeout
        remaining = bins - item
        valid = remaining >= 0
        if not np.any(valid):
            return np.full(len(bins), -np.inf)
        with np.errstate(divide='ignore', invalid='ignore'):
            score = np.where(valid, -remaining, -np.inf)
        return score
    
    remaining = bins - item
    valid = remaining >= 0
    
    if not np.any(valid):
        return np.full(len(bins), -np.inf)
    
    # Primary component: Relative Best-Fit
    # Minimize remaining/item to normalize gap by current item size
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_gap = remaining / max(item, 1e-10)
    
    # Secondary component: Smooth penalty for large gaps
    # For Weibull with shape ~1.5-2.0 and scale ~10-20, typical items are 5-30
    # Gaps > 40 are likely unusable. Use a smooth step-like penalty.
    # This approximates a soft-min: penalty grows smoothly as gap exceeds threshold
    gap_threshold = 35.0  # Gaps above this are increasingly likely wasted
    gap_width = 15.0      # Smoothness of the transition
    
    # Smooth penalty: 0 for small gaps, grows smoothly for large gaps
    # Using tanh for smooth saturation
    with np.errstate(over='ignore'):
        smooth_penalty = np.tanh((remaining - gap_threshold) / gap_width)
    
    # Combine: primary score minus penalty for large gaps
    # We want to minimize both relative gap and absolute large gaps
    with np.errstate(over='ignore', invalid='ignore'):
        score = np.where(valid, -(relative_gap + 0.3 * smooth_penalty), -np.inf)
    
    return score
# EVOLVE-BLOCK-END
