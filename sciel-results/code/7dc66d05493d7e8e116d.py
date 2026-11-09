import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Fragmentation-aware bin packing heuristic.
    
    Scores bins based on the utility of the remaining space after placing the item.
    Instead of simply preferring the tightest fit (Best Fit), this model evaluates
    whether the leftover space is "useful" for future items.
    
    Key insight: A remaining space is useful if it can fit a reasonable fraction
    of the current item size (suggesting similar-sized future items might fit),
    but not so large that it represents significant waste.
    
    The score combines:
    1. A base preference for smaller remaining space (efficiency)
    2. A "fragmentation penalty" for remaining spaces that are too small to be useful
       (below ~15% of item size) or too large (wasteful, above ~80% of item size)
    3. A "useful gap" bonus for remaining spaces in the sweet spot (15-60% of item)
    """
    remaining = bins - item
    
    # Initialize scores with -inf for invalid bins
    scores = np.full_like(remaining, -np.inf, dtype=np.float64)
    
    # Only consider bins with enough remaining capacity
    valid = remaining >= 0
    
    if not np.any(valid):
        return scores
    
    r = remaining[valid]
    
    # Normalize by item size to make the heuristic scale-invariant
    # ratio = remaining / item_size
    # ratio = 0 means perfect fit
    # ratio = 1 means remaining space equals item size
    ratio = r / (item + 1e-10)
    
    # Component 1: Base efficiency score - prefer smaller remaining
    # This is the "best fit" component, but scaled
    base_score = -ratio
    
    # Component 2: Fragmentation penalty
    # Penalize remaining spaces that are too small to be practically useful
    # A gap less than ~15% of item size is likely to become "dead space"
    # that can't be filled by any reasonable future item
    small_gap_threshold = 0.15
    small_gap_penalty = np.where(ratio < small_gap_threshold,
                                  (small_gap_threshold - ratio) / small_gap_threshold,
                                  0.0) * 2.0
    
    # Penalize very large gaps that represent significant waste
    # If remaining > 80% of item size, we're not packing efficiently
    large_gap_threshold = 0.80
    large_gap_penalty = np.where(ratio > large_gap_threshold,
                                  (ratio - large_gap_threshold) / large_gap_threshold,
                                  0.0) * 1.5
    
    # Component 3: Useful gap bonus
    # Reward remaining spaces in the "sweet spot" where they're likely to be
    # filled by future items of similar or smaller size
    # Sweet spot: 15% to 60% of item size
    sweet_low = 0.15
    sweet_high = 0.60
    
    # Gaussian-like bump centered in the sweet spot
    sweet_center = (sweet_low + sweet_high) / 2.0  # 0.375
    sweet_sigma = (sweet_high - sweet_low) / 3.0   # ~0.15
    
    useful_gap_bonus = np.exp(-((ratio - sweet_center) ** 2) / (2.0 * sweet_sigma ** 2)) * 1.0
    
    # Combine all components
    # Weights: base efficiency is primary, fragmentation penalties are secondary,
    # useful gap bonus provides fine-tuning
    weight_base = 1.0
    weight_small_penalty = 1.0
    weight_large_penalty = 1.0
    weight_useful_bonus = 0.5
    
    combined = (weight_base * base_score 
                - weight_small_penalty * small_gap_penalty 
                - weight_large_penalty * large_gap_penalty 
                + weight_useful_bonus * useful_gap_bonus)
    
    scores[valid] = combined
    return scores
