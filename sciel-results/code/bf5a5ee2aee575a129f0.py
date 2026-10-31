"""Rewritten online bin packing priority function with polarization heuristic."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    
    Strategy: Polarization-based scoring
    - Strongly prefer bins that are already quite full (to finish them off).
    - Avoid bins that are nearly empty (to keep them available for large items later).
    - Specifically, we want to minimize the waste of opening a new bin or leaving a bin half-full.
    """
    # Get remaining capacities
    remaining = bins - item
    
    # Only consider bins with enough capacity
    valid_mask = remaining >= 0
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid_mask):
        return scores
    
    valid_remaining = remaining[valid_mask]
    valid_bins = bins[valid_mask]
    
    # Compute fill ratio for valid bins (0 to 1)
    fill_ratio = valid_bins / 100.0  # bins represent used capacity, capacity is 100
    
    # Compute the new fill ratio after placing the item
    new_fill_ratio = (valid_bins + item) / 100.0
    
    # Polarization score:
    # We want to either fill bins close to 100% or keep them near 0%.
    # A bin is "good" if it's already close to full (finish it) or very empty (keep it for big items).
    
    # Score component 1: Prefer bins that will become very full (close to 100%)
    # This encourages finishing bins
    fullness_score = new_fill_ratio ** 4  # Strong preference for nearly full bins
    
    # Score component 2: Penalty for leaving bins in the "middle" zone (20%-80%)
    # This is the hardest zone for packing
    middle_penalty = np.exp(-((new_fill_ratio - 0.5) ** 2) / 0.05)  # Gaussian penalty centered at 0.5
    
    # Score component 3: Prefer bins that are already well-used (avoid opening new bins unnecessarily)
    # But not so much that we ignore the polarization benefit
    usage_score = fill_ratio ** 2
    
    # Combine: High score for bins that either finish off nearly-full bins or are very empty
    # Low score for bins that end up in the middle zone
    combined_score = fullness_score - 0.5 * middle_penalty + 0.3 * usage_score
    
    # Additional: Prefer bins where the item fills a large fraction of the remaining space
    # This reduces fragmentation
    if np.any(valid_remaining > 0):
        fill_fraction = np.where(valid_remaining > 0, item / valid_remaining, 0)
        # Prefer items that fill most of the remaining space
        fill_fraction_score = fill_fraction ** 3
        combined_score += 0.2 * fill_fraction_score
    
    # Assign scores to valid bins
    scores[valid_mask] = combined_score
    
    return scores
# EVOLVE-BLOCK-END
