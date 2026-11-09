"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin using a nonlinear fragmentation-aware heuristic combined
    with a 'Next-Item Fit' proxy.
    
    The primary score uses a power-law transform on the normalized remaining
    capacity to prefer tighter fits (minimizing fragmentation). The secondary
    score penalizes bins whose remaining capacity is far from the median item
    size of the sequence, acting as a proxy for how well the bin can accommodate
    typical future items without creating unusable gaps.
    
    Score = -(rem_norm)^alpha - beta * ((rem_norm - median_norm)^2)
    
    Where:
    - rem_norm = remaining / capacity
    - alpha = 0.7 (tight fit preference)
    - beta = 0.5 (weight of next-item fit penalty)
    - median_norm = median item size / capacity (estimated from the current item and bins)
    """
    start_time = time.time()
    
    # Compute remaining capacity if item is placed in each bin
    remaining = bins - item
    
    # Only consider bins with enough remaining capacity
    valid_mask = remaining >= 0
    
    # Create scores array, default to very low for invalid bins
    scores = np.full(len(bins), -np.inf, dtype=np.float64)
    
    if not np.any(valid_mask):
        return scores
    
    rem_valid = remaining[valid_mask]
    
    # Capacity is 100 per the task description
    capacity = 100.0
    
    # Normalize remaining by capacity
    rem_norm = rem_valid / capacity
    
    # Estimate the median item size.
    # Since we are online, we don't know future items. However, the distribution
    # is Weibull with known parameters in the generator, but here we only have
    # the current item and the state of bins.
    # A robust proxy for "typical" item size is the current item itself, or
    # perhaps a smoothed average of items that have been placed.
    # But we don't have history.
    # The directive suggests: "median item size of the entire sequence".
    # We cannot access the entire sequence.
    # However, we can approximate the median of the Weibull distribution.
    # The task says "five generated Weibull sequences".
    # Standard Weibull parameters for bin packing tasks are often shape=1.5, scale=25
    # or similar. Let's assume a typical median.
    # Actually, a safer heuristic is to use the current item as a proxy for the
    # "typical" item, or to use a fixed value based on the capacity.
    # Let's use the current item as a strong signal for the next item's size,
    # assuming some autocorrelation or stationarity.
    # Or, more simply, the "Next-Item Fit" proxy suggests we want the remaining
    # space to be close to the size of the *next* item.
    # Since we don't know the next item, using the current item as a proxy is
    # a reasonable first-order approximation.
    # Let's use the current item as the target remaining size.
    target_rem = item  # Proxy for next item size
    
    # Normalize target
    target_norm = target_rem / capacity
    
    # Primary: nonlinear tight-fit preference
    alpha = 0.7
    primary = -(rem_norm ** alpha)
    
    # Secondary: Next-Item Fit penalty
    # Penalize deviation from the target remaining space
    beta = 0.5
    secondary = -beta * ((rem_norm - target_norm) ** 2)
    
    # Combined score
    scores[valid_mask] = primary + secondary
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 140:  # Safety margin
        return scores
    
    return scores
# EVOLVE-BLOCK-END
