"""Fragmentation-aware online bin packing heuristic."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for placing `item`. Higher score = better placement.
    
    Uses a refined fragmentation-aware model:
    1. Base cost: Remaining capacity (Best Fit tendency).
    2. Fragmentation penalty: Sharp penalty for leaving 'dead zone' remaining space 
       (space too small to be useful but too large to be considered full).
    3. Ratio penalty: Penalizes placing small items in large bins (low fill ratio),
       which wastes capacity that could have been used for larger items.
    
    The goal is to minimize total waste (remaining capacity) while avoiding 
    creating unusable fragments and maintaining flexibility for future items.
    """
    # Remaining capacity in each bin after placing item
    remaining = bins - item
    
    # Only consider bins with enough remaining capacity
    valid = remaining >= 0
    
    # For invalid bins, assign a very low score
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid):
        return scores
    
    rem_valid = remaining[valid]
    bin_caps_valid = bins[valid]
    
    # 1. Base Cost: Linear penalty for remaining space.
    # We want to minimize remaining space (Best Fit).
    base_cost = rem_valid
    
    # 2. Fragmentation Penalty:
    # We identify a "dead zone" of remaining capacity where the space is 
    # likely too small to fit typical future items (Weibull distributed).
    # For capacity 100 and Weibull items, mean size is often ~10-15.
    # A remaining space of ~5-15 is often a bad fragment.
    # We use a Gaussian penalty centered at mu_frag with sigma_frag.
    
    mu_frag = 8.0   # Center of the bad fragment zone
    sigma_frag = 4.0 # Width of the bad zone
    
    # Calculate squared distance from the bad zone center
    diff = rem_valid - mu_frag
    frag_penalty = np.exp(-(diff ** 2) / (2 * sigma_frag ** 2))
    
    # Weight of the fragmentation penalty
    # We want this to be significant but not dominate the base cost entirely
    # unless the fit is in the bad zone.
    w_frag = 10.0
    
    # 3. Ratio Penalty:
    # Penalize placements where the item is small relative to the bin capacity.
    # This encourages filling bins with larger items or keeping bins for larger items.
    # ratio = item / bin_capacity
    # If ratio is very low (e.g., < 0.1), it's a bad use of space.
    # We use a Gaussian penalty centered at a low ratio (e.g., 0.05).
    
    ratios = item / bin_caps_valid
    mu_ratio = 0.05  # Center of "too small" ratio
    sigma_ratio = 0.05
    
    ratio_diff = ratios - mu_ratio
    ratio_penalty = np.exp(-(ratio_diff ** 2) / (2 * sigma_ratio ** 2))
    
    w_ratio = 5.0
    
    # Combined Cost
    # We sum the penalties. Higher cost = worse placement.
    # Score is negative cost, so higher score = better placement.
    cost = base_cost + w_frag * frag_penalty + w_ratio * ratio_penalty
    
    # Assign scores to valid bins
    scores[valid] = -cost
    
    return scores
# EVOLVE-BLOCK-END
