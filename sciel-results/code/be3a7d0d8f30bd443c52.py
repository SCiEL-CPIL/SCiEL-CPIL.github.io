import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining_after = bins - item
    scores = np.full(len(bins), -np.inf, dtype=float)
    
    valid_mask = remaining_after >= 0
    if not np.any(valid_mask):
        return scores
        
    r_new = remaining_after[valid_mask]
    b_valid = bins[valid_mask]
    
    # Base score: Best-Fit (prefer smaller remaining capacity)
    # This encourages tight packing.
    base_scores = -r_new
    
    # Fragmentation-aware penalty based on ratio
    # Ratio of item size to available capacity (bins)
    # We want to penalize bins where the item occupies a moderate fraction (20-50%)
    # of the available space, as this may leave a "medium" fragment that is
    # less useful than either a very small (tight) or very large (loose) one.
    
    # Calculate ratio: item / bins
    # Avoid division by zero (though bins should be > 0 for valid items if item > 0)
    # If bins is 0, it wouldn't be valid for positive item.
    ratios = item / b_valid
    
    penalty = np.zeros_like(r_new)
    
    # Define the penalty zone for the ratio
    lower_ratio = 0.2
    upper_ratio = 0.5
    
    in_zone = (ratios >= lower_ratio) & (ratios <= upper_ratio)
    
    if np.any(in_zone):
        # Create a penalty that is strongest in the middle of the zone
        # and tapers off at the edges.
        # Center of zone
        center_ratio = (lower_ratio + upper_ratio) / 2.0
        half_width = (upper_ratio - lower_ratio) / 2.0
        
        # Distance from center, normalized
        dist = np.abs(ratios[in_zone] - center_ratio)
        norm_dist = dist / half_width
        
        # Penalty is max at center, 0 at edges
        # Max penalty value needs to be significant relative to base scores.
        # Base scores are -r_new. r_new can be up to 100.
        # A penalty of 10-20 should be noticeable.
        max_penalty = 15.0
        p_val = max_penalty * (1 - norm_dist)
        
        penalty[in_zone] = p_val
        
    final_scores = base_scores - penalty
    scores[valid_mask] = final_scores
    
    return scores
