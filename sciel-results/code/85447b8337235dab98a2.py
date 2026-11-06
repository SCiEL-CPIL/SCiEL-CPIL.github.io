"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for online bin packing using a smooth surrogate approach.
    Uses soft-min to prefer bins with minimal remaining capacity while
    incorporating fill-ratio information for better packing efficiency.
    """
    rem = bins - item
    
    # Mask for valid bins (enough capacity)
    valid = rem >= 0
    
    # For invalid bins, assign very low score
    scores = np.full_like(rem, -np.inf, dtype=float)
    
    if not np.any(valid):
        return scores
    
    valid_rem = rem[valid]
    valid_bins = bins[valid]
    
    # Normalize remainder by item size for scale invariance
    norm_rem = valid_rem / max(item, 1e-10)
    
    # Component 1: Soft-min based fit score
    # Use log-sum-exp to create a smooth approximation of min
    # Lower normalized remainder should get higher score
    # We want to reward bins where the item fits tightly
    # Use negative exp to create a smooth penalty for large remainders
    # This creates a smooth surrogate that approximates best-fit behavior
    
    # Temperature parameter for soft-min (lower = sharper, more like hard min)
    temp = 0.1
    # Compute soft-min score: prefer bins with smallest remainder
    # Use -exp(-norm_rem/temp) so that smaller norm_rem gives higher score
    soft_fit = -np.exp(-norm_rem / temp)
    
    # Component 2: Fill ratio component
    # Prefer bins that are already partially filled (higher fill ratio)
    # This helps consolidate items and reduce waste
    fill_ratio = item / np.maximum(valid_bins, 1e-10)
    
    # Component 3: Quadratic penalty for extreme remainders
    # Penalize both very small remainders (wasted space) and very large ones
    # Optimal remainder is around 0.3-0.5 of item size
    optimal_rem = 0.4 * item
    quad_penalty = -0.1 * ((valid_rem - optimal_rem) / max(item, 1e-10)) ** 2
    
    # Combined score with weighted components
    w_soft = 1.0
    w_fill = 0.5
    w_quad = 0.3
    
    combined = w_soft * soft_fit + w_fill * fill_ratio + w_quad * quad_penalty
    
    scores[valid] = combined
    
    return scores
# EVOLVE-BLOCK-END
