"""Rewritten program for ant colony bin packing heuristic design."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    """
    Compute an item-by-item attractiveness matrix for placing item j right after
    item i in the same bin. Larger values indicate more attractive placements.
    
    Strategy: We want to encourage placements that leave a well-shaped residual
    capacity in the bin, so that subsequent items can fit. A good heuristic
    promotes pairs (i, j) where:
      1. demands[i] + demands[j] <= capacity (feasibility)
      2. The residual capacity (capacity - demands[i] - demands[j]) is such that
         it can accommodate other common item sizes.
    
    We use a combination of:
      - Feasibility mask (infeasible pairs get 1e-9)
      - Residual utility: prefer residuals that are "useful" - not too small
        (wasted space) and not too large (could have fit a bigger item)
      - Size complementarity: prefer pairs where the two items are of similar
        or complementary sizes that pack well together.
    
    Specifically, we define:
      residual = capacity - demands[i] - demands[j]
      residual_score = 1 / (1 + |residual - target_residual|)
      where target_residual is chosen to be a moderate value.
    
    Additionally, we add a bonus for pairs where both items are relatively large
    (since packing large items together reduces the number of bins needed).
    """
    demands = np.asarray(demands, dtype=np.float64)
    n = len(demands)
    
    # Pairwise sums
    pair_sum = demands[:, None] + demands[None, :]
    
    # Feasibility: pair must fit
    feasible = pair_sum <= capacity
    
    # Residual capacity
    residual = capacity - pair_sum
    
    # Target residual: we want the residual to be in a useful range.
    # If residual is too small (< 20), it's wasted space.
    # If residual is too large (> 80), we might have been able to fit a bigger item.
    # A good target is around 40-60, where many items (20-99) could still fit.
    # Let's use a Gaussian-like score centered at a moderate residual.
    
    # Score based on residual: prefer residuals around 30-50
    # Use a smooth function: score = exp(-((residual - 40)^2) / (2 * sigma^2))
    sigma = 25.0
    residual_score = np.exp(-((residual - 40.0) ** 2) / (2 * sigma ** 2))
    
    # Bonus for packing larger items together
    # Normalize demands to [0, 1]
    d_min = 20.0
    d_max = 99.0
    d_norm = (demands - d_min) / (d_max - d_min)
    d_norm = np.clip(d_norm, 0, 1)
    
    # Pairwise product of normalized sizes: encourages pairing large items
    size_bonus = d_norm[:, None] * d_norm[None, :]
    
    # Combine: main signal is residual utility, secondary is size complementarity
    # Weight: 0.7 for residual, 0.3 for size
    heuristic = 0.7 * residual_score + 0.3 * size_bonus
    
    # Ensure positive values for feasible pairs
    heuristic = np.where(feasible, heuristic, 1e-9)
    
    # Clamp to ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
