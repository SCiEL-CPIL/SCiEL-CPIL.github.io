"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    pair = d[:, None] + d[None, :]
    mask = pair <= capacity
    # Heuristic: favor pairs that nearly fill the bin, with stronger preference
    # for larger items j (since they consume bin capacity more efficiently)
    # and moderate penalty for very small pairs (waste).
    # Use a combination: (pair/capacity)^power * max(demand_j, 1)
    # This makes large items more attractive when they fit tightly.
    ratio = np.where(mask, pair / capacity, 0.0)
    # Strongly favor high fill ratios
    fill_score = ratio ** 12
    # Boost by the size of item j (column index) to prefer larger items
    size_boost = d[None, :] / 100.0  # normalized to [0.2, 0.99]
    heuristic = fill_score * (1.0 + size_boost)
    # Ensure non-positive/infeasible entries become 1e-9
    heuristic = np.where(mask & (heuristic > 0), heuristic, 1e-9)
    return heuristic
# EVOLVE-BLOCK-END
