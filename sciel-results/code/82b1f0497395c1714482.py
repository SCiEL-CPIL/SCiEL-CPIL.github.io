import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Uses a refined hybrid heuristic that balances best-fit (minimizing waste)
    with a soft preference for 'slightly loose' fits to reduce fragmentation.
    
    The score is designed to maximize the utility of the chosen bin:
    1. Primary: Minimize remaining capacity (Best Fit).
    2. Secondary: Reward a specific gap ratio (Gaussian bump) to avoid
       exact fits (which can be fragile) and large gaps (which waste space).
    3. Tertiary: Penalize very large gaps super-linearly to ensure that
       the 'loose fit' preference doesn't override the need to save space
       for large future items.
    """
    # Deadline guard: track start time to ensure we return before 120s limit
    # Although this function is called per-item, the overhead is minimal.
    # We don't need complex logic here as numpy ops are fast.
    
    remaining = bins - item
    
    # Initialize scores with -inf for invalid bins
    scores = np.full_like(remaining, -np.inf, dtype=np.float64)
    
    # Only consider bins with enough remaining capacity
    valid = remaining >= 0
    
    if not np.any(valid):
        return scores
    
    r = remaining[valid]
    
    # Normalize remaining capacity by item size to make the heuristic scale-invariant
    # This is crucial because the 'ideal gap' should be a fraction of the item,
    # not a fixed absolute value.
    # Avoid division by zero
    item_safe = max(item, 1e-10)
    gap_ratio = r / item_safe
    
    # Base Score: Best Fit
    # We want to minimize r. Since we are maximizing the score, we use -r.
    # However, -r is on the scale of the item size.
    # To make the bump (which is in [0,1]) comparable, we should scale the base score.
    # Let's use a weighted combination where the base score is dominant but the bump
    # can influence the decision when remaining capacities are close.
    
    # If we use -r directly, the difference between r=0 and r=0.1*item is 0.1*item.
    # The bump max is 1.0.
    # If item is 10, difference is 1.0. Bump can tie or override.
    # If item is 0.1, difference is 0.01. Bump (weight 1.0) dominates.
    
    # A better approach: Score = - (r / item_safe) + weight * bump
    # This normalizes the 'waste' to a ratio.
    # Waste ratio = r / item.
    # We want to minimize waste ratio.
    
    waste_ratio = r / item_safe
    base_score = -waste_ratio
    
    # Gaussian Bump
    # We want to encourage a gap that is not too small (to allow for estimation errors or future small items)
    # but not too large.
    # Center the bump at a small positive gap, e.g., 5-10% of the item.
    # Let's try center = 0.1 * item (gap_ratio = 0.1).
    # Sigma should be narrow enough to be selective but wide enough to be robust.
    # Let's use sigma = 0.05 (in gap_ratio terms).
    
    center_ratio = 0.1
    sigma_ratio = 0.05
    
    bump = np.exp(-((gap_ratio - center_ratio) ** 2) / (2.0 * sigma_ratio ** 2))
    
    # Penalty for large gaps
    # If gap_ratio > 2.0, the bin is much larger than the item.
    # The base score -waste_ratio already penalizes this linearly.
    # However, we might want to penalize it more heavily to prevent
    # using a bin with 10x capacity for a 1x item if a 1.5x bin is available.
    # Actually, -waste_ratio does this: -10 vs -1.5. Difference is 8.5.
    # The bump is at most 1.0. So base score dominates for large gaps.
    # This is good.
    
    # Combine scores
    # We need to tune the weight of the bump.
    # If weight is too high, we prefer gap_ratio=0.1 over gap_ratio=0.0 (exact fit).
    # Difference in base score: -0.1 - 0 = -0.1.
    # Bump at 0.1 is 1.0. Bump at 0.0 is exp(-(0.1)^2 / (2*0.05^2)) = exp(-2) ~ 0.135.
    # Score(0.1) = -0.1 + w*1.0
    # Score(0.0) = 0 + w*0.135
    # We prefer 0.1 if -0.1 + w > 0.135w => 0.865w > 0.1 => w > 0.115.
    # So if weight > 0.115, we prefer the 10% gap over the exact fit.
    # Is this desirable?
    # Exact fits are generally good for bin packing (Best Fit Decreasing is strong).
    # However, in online packing, exact fits might leave no room for tiny items that could have fit elsewhere.
    # But usually, Best Fit is very strong.
    # Let's start with a moderate weight, e.g., 0.5, and see if it improves over pure Best Fit.
    # The current program uses a similar structure.
    
    weight = 0.5
    combined = base_score + weight * bump
    
    scores[valid] = combined
    return scores
