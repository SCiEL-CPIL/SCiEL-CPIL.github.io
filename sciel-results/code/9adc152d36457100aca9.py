"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Uses a composite score based on:
    1. Best-fit: prefer bins that leave least remaining capacity
    2. Tightness: prefer bins that are already well-utilized
    3. Avoid creating bins that are too empty relative to item size
    """
    capacity = 100.0
    
    # Remaining capacity after placing item
    remaining = bins - item
    
    # Only consider bins with enough capacity
    valid_mask = remaining >= 0
    
    # Initialize scores to very negative (invalid bins)
    scores = np.full_like(bins, -1e18, dtype=float)
    
    if not np.any(valid_mask):
        return scores
    
    valid_remaining = remaining[valid_mask]
    valid_bins = bins[valid_mask]
    
    # Feature 1: Best-fit score (less remaining = better)
    # Normalize: less remaining is better, so negative remaining
    best_fit_score = -valid_remaining
    
    # Feature 2: Utilization of the bin before placing item
    # Higher utilization = better (reduces fragmentation)
    utilization = valid_bins / capacity
    
    # Feature 3: Post-placement utilization
    post_utilization = (valid_bins + item) / capacity
    
    # Feature 4: How "tight" the fit is - prefer fits where item is a significant portion
    # of the remaining space (avoids tiny items in large bins)
    tightness = np.where(valid_remaining > 1e-9, item / (valid_remaining + item), 1.0)
    
    # Composite score:
    # - Primary: best-fit (minimize leftover)
    # - Secondary: prefer bins with higher pre-placement utilization
    # - Tertiary: tightness bonus
    weight_best_fit = 1.0
    weight_utilization = 0.3
    weight_tightness = 0.2
    
    composite = (weight_best_fit * best_fit_score + 
                 weight_utilization * utilization + 
                 weight_tightness * tightness)
    
    scores[valid_mask] = composite
    
    return scores
# EVOLVE-BLOCK-END
