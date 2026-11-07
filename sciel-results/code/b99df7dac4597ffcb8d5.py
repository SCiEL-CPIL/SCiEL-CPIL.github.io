"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    W = np.asarray(demands, dtype=np.float64)
    n = len(W)
    
    # Pair sums and fit mask
    pair_sums = W[:, None] + W[None, :]
    fit_mask = pair_sums <= capacity
    
    # Remaining capacity after placing j after i
    rem = capacity - pair_sums  # shape (n, n)
    rem = np.maximum(rem, 0.0)
    
    # For each pair (i,j) that fits, compute heuristic score
    
    # Component 1: Tight fit reward - prefer small remaining space
    # Use 1/(rem + eps) so tighter fits get higher scores
    eps = 1e-6
    tight_fit = 1.0 / (rem + eps)
    tight_fit = np.where(fit_mask, tight_fit, 0.0)
    
    # Component 2: Prefer placing larger items (they're harder to place later)
    # Normalize by capacity
    item_size_score = W[None, :] / capacity  # shape (n, n), broadcast over i
    item_size_score = np.where(fit_mask, item_size_score, 0.0)
    
    # Component 3: Penalize wasting space that could fit a large remaining item
    # For each j, compute the maximum item size that could fit in rem
    # If rem is large but no large items remain, less penalty
    # Simple proxy: use the max demand as a reference
    max_demand = W.max()
    # If rem is close to max_demand, it's "wasteful" if we could have placed a big item
    waste_penalty = np.where(fit_mask, rem / max_demand, 0.0)
    
    # Combine: weight tight fit heavily, add size preference, subtract waste
    # Scale tight_fit to be in a reasonable range
    tf_max = tight_fit.max()
    if tf_max > 0:
        tight_fit_norm = tight_fit / tf_max
    else:
        tight_fit_norm = tight_fit
    
    # Final heuristic: emphasize tight fits, prefer larger items, penalize waste
    H = tight_fit_norm + 0.5 * item_size_score - 0.3 * waste_penalty
    
    # Ensure non-negative, set non-fitting to 1e-9
    H = np.where(fit_mask, np.maximum(H, 1e-9), 1e-9)
    
    # Ensure finite values
    H = np.where(np.isfinite(H), H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
