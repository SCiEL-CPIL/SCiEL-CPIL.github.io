import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Uses a hybrid heuristic combining Best Fit with a Gaussian bump
    centered at a small fraction of the item size to encourage
    'slightly loose' fits that may reduce fragmentation over time.
    Includes a mild penalty for very large gaps to avoid waste.
    """
    remaining = bins - item
    
    # Initialize scores with -inf for invalid bins
    scores = np.full_like(remaining, -np.inf, dtype=np.float64)
    
    # Only consider bins with enough remaining capacity
    valid = remaining >= 0
    
    if not np.any(valid):
        return scores
    
    r = remaining[valid]
    
    # Primary: Best Fit (smaller remaining is better)
    base_score = -r
    
    # Secondary: Gaussian bump for "slightly loose" fits
    # Center at 0.1 * item, Sigma 0.05 * item
    # This encourages leaving a small gap that can be filled by future small items
    center = 0.1 * item
    sigma = 0.05 * item + 1e-10
    bump = np.exp(-((r - center) ** 2) / (2.0 * sigma ** 2))
    
    # Tertiary: Mild penalty for very large gaps (waste)
    # This is a refinement from Inspiration 2, but with a much smaller weight
    # to not override the best-fit preference.
    gap_ratio = r / (item + 1e-10)
    # Only penalize gaps that are significantly larger than the item
    # Start penalty at gap_ratio > 3.0, with a gentle slope
    large_gap_penalty = np.maximum(gap_ratio - 3.0, 0.0) * 0.05
    
    # Combine: best fit is dominant, bump provides fine-tuning,
    # large gap penalty is a gentle nudge
    weight_bump = 0.5
    combined = base_score + weight_bump * bump - large_gap_penalty
    
    scores[valid] = combined
    return scores
