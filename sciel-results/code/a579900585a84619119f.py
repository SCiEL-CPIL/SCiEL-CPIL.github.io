import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # bins represents remaining capacity in each bin (capacity - used)
    remaining = bins
    valid = remaining >= item
    
    if not np.any(valid):
        return np.full(len(bins), -np.inf)
    
    # Weibull distributions often have a mix of small and large items.
    # A common heuristic is to prefer bins that are "medium" full,
    # avoiding both very full bins (risk of fragmentation) and very empty bins (waste).
    # Let's try a score that peaks at some intermediate fill level.
    
    # Normalize remaining capacity to [0, 1] relative to the max remaining
    max_rem = np.max(remaining[valid]) if np.any(valid) else 1.0
    if max_rem == 0:
        return np.where(valid, 0.0, -np.inf)
    
    # Fill level: 1.0 means empty (max remaining), 0.0 means full (min remaining among valid)
    # Actually, let's think in terms of "used" relative to capacity.
    # But we don't know capacity directly, only remaining.
    # Let's use the remaining capacity directly.
    
    # Strategy: Prefer bins with remaining capacity close to the item size?
    # Or prefer bins with a "balanced" remaining capacity?
    
    # Let's try: score = -|remaining - target_remaining|
    # What is a good target? Maybe the median of valid remaining capacities?
    valid_rems = remaining[valid]
    if len(valid_rems) > 0:
        target = np.median(valid_rems)
    else:
        target = 0.0
    
    # Score is negative distance from target. Closer to target is better.
    # This creates a "balanced" packing where bins tend to have similar remaining capacity.
    scores = np.where(valid, -(np.abs(remaining - target)), -np.inf)
    
    return scores
