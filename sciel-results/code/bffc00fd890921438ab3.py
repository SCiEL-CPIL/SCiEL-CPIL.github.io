"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    """
    Compute an item-pair attractiveness matrix for ACO bin packing.
    
    The heuristic encodes how attractive it is to place item j right after item i
    in the same bin. Larger values mean more attractive.
    
    Design: We want to encourage placements that leave "good" remaining space.
    A pair (i, j) is attractive if:
    1. They fit together (demands[i] + demands[j] <= capacity)
    2. The remaining space after placing both is useful (not too small, not too large)
    
    We use a smooth surrogate: attractiveness peaks when the combined size is a
    significant fraction of capacity (good packing density) but not so large that
    it leaves very little room for future items.
    
    Specifically, we score pair (i,j) as:
      - 0 (or 1e-9) if they don't fit
      - Otherwise, a function that rewards moderate-to-high combined sizes
    
    We use: f(s) = s/capacity * exp(-alpha * (s/capacity - beta)^2)
    where s = demands[i] + demands[j], and we tune alpha, beta to favor
    combined sizes around 0.7-0.8 of capacity.
    
    Actually, let me think more carefully. The ACO picks items with probability
    proportional to pheromone[i,j] * heuristic[i,j]. Initially pheromone is 1
    everywhere, so the heuristic dominates early on.
    
    A good heuristic should:
    - Strongly prefer pairs that fit
    - Among fitting pairs, prefer those that pack efficiently
    
    Let me use: for fitting pairs, attractiveness = (combined/capacity)^gamma
    where gamma > 1 emphasizes larger combined sizes (better utilization).
    
    But we also don't want to over-penalize small items that could still be useful.
    Let's use a combination: base score based on utilization, with a small
    penalty for leaving too little or too much space.
    
    Simple effective approach: attractiveness = (d_i + d_j) / capacity for fitting pairs,
    raised to a power to emphasize larger combinations.
    """
    d = np.asarray(demands, dtype=np.float64)
    n = len(d)
    
    # Combined sizes
    pair_sum = d[:, None] + d[None, :]
    
    # Mask: only pairs that fit
    fits = pair_sum <= capacity
    
    # Base attractiveness: utilization-based
    # We want to reward pairs that use a good fraction of capacity
    # Use (pair_sum / capacity) as base, which ranges from ~0.27 to 1.0 for fitting pairs
    # Raise to a power > 1 to emphasize larger combined sizes
    base = np.zeros((n, n), dtype=np.float64)
    base[fits] = (pair_sum[fits] / capacity) ** 1.5
    
    # Additionally, penalize pairs where the sum is very close to capacity
    # (leaves no room) or very small (wastes space)
    # Actually, for bin packing, using more space is generally better
    # unless it blocks future placements. The ACO will handle the blocking
    # via pheromone feedback.
    
    # For non-fitting pairs, set to 1e-9
    result = np.where(fits, base, 1e-9)
    
    # Ensure diagonal is handled (item with itself) - these are valid if 2*d[i] <= capacity
    # They're already handled by the mask
    
    # Ensure all values are positive and finite
    result = np.where(np.isfinite(result) & (result > 0), result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
