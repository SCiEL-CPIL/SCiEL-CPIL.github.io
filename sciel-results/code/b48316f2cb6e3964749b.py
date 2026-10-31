"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity using a probabilistic risk model.
    
    The approach models bin selection as a risk assessment:
    - Compute the global statistical trend of remaining capacities across all open bins.
    - Score each candidate bin based on how its remaining capacity after placing the item
      aligns with the global distribution, penalizing bins that deviate significantly
      from the mean (high variance contribution).
    - Additionally, incorporate a fit component to avoid leaving large unusable gaps.
    
    Returns an array of scores; the item goes to the highest-scoring bin.
    """
    remaining = bins - item
    
    # Only consider bins with enough remaining capacity
    valid_mask = remaining >= 0
    
    # Get the remaining capacities of valid bins
    valid_remaining = remaining[valid_mask]
    
    # If no valid bins, return all negative infinity (shouldn't happen in practice)
    if len(valid_remaining) == 0:
        return np.full(len(bins), -np.inf)
    
    # Compute global statistics of remaining capacities across ALL bins (not just valid)
    # This captures the global load distribution
    all_remaining = bins
    global_mean = np.mean(all_remaining)
    global_var = np.var(all_remaining)
    
    # For each valid bin, compute a score based on:
    # 1. How well the post-placement remaining capacity aligns with the global mean
    #    (lower deviation = better, as it maintains balance)
    # 2. A fit component: prefer bins that leave less remaining capacity (best-fit)
    #    but not too much, to avoid fragmentation
    
    # Deviation from global mean after placement
    deviation = valid_remaining - global_mean
    
    # Negative squared deviation (we want to minimize variance contribution)
    variance_score = -deviation ** 2
    
    # Fit score: prefer bins where the item fits tightly
    # Use a softer penalty for large remaining capacities
    # Normalize by the capacity (100) to make it scale-invariant
    capacity = 100.0
    fit_score = -(valid_remaining / capacity) ** 2
    
    # Combine scores: weight the variance alignment and fit
    # The variance score encourages bins that keep the global distribution tight
    # The fit score encourages tight packing
    combined_score = variance_score + 0.3 * fit_score
    
    # Create the full score array
    scores = np.full(len(bins), -np.inf)
    scores[valid_mask] = combined_score
    
    return scores
# EVOLVE-BLOCK-END
