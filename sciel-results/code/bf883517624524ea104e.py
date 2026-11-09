"""Combined approach: tight-fill heuristic with size-aware weighting and smooth surrogate."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 315  # safety margin before 330s kill


def heuristics(demands, capacity):
    """
    Compute the item-pair attractiveness matrix for ACO bin packing.
    
    Strategy: Combine tight-fill preference with size-aware weighting.
    - Favor pairs where size(i) + size(j) is close to capacity (tight packing)
    - Weight by size of item j (larger items are harder to place, so prioritize)
    - Use smooth Gaussian-like kernel for the tight-fill score to avoid discontinuities
    - Ensure feasibility: only pairs where size(i) + size(j) <= capacity get high scores
    """
    d = np.asarray(demands, dtype=np.float64)
    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    
    # Time guard at start
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: simple ratio-based heuristic
        pair = d[:, None] + d[None, :]
        mask = pair <= capacity
        ratio = np.where(mask, pair / capacity, 0.0)
        return np.where(mask, ratio ** 8, 1e-9)
    
    # Compute pair sums
    pair_sum = d[:, None] + d[None, :]
    
    # Feasibility mask: pair must fit in capacity
    feasible = pair_sum <= capacity
    
    # Remaining capacity after placing item i
    remaining = capacity - d[:, None]
    
    # Tight-fill score: Gaussian kernel peaking at pair_sum == capacity
    # Use a moderate sigma to balance specificity vs. robustness
    sigma = capacity * 0.15  # 15% of capacity as width parameter
    gap = pair_sum - capacity
    tight_score = np.exp(-0.5 * (gap / sigma) ** 2)
    
    # Size-aware weighting: prefer larger item j
    # Normalize to [0.2, 1.0] range for stability
    size_boost = d[None, :] / 100.0
    size_boost = np.clip(size_boost, 0.2, 1.0)
    
    # Combine: tight-fill score weighted by size of j
    # The product peaks when pair_sum is close to capacity AND j is large
    heuristic = tight_score * size_boost
    
    # For infeasible pairs, set to very small value
    heuristic = np.where(feasible, heuristic, 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    # Final time guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return heuristic
    
    return heuristic
# EVOLVE-BLOCK-END
