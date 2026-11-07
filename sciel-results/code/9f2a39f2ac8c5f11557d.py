"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    demands = np.asarray(demands, dtype=float)
    n = demands.shape[0]
    
    # Pair sum matrix
    pair_sum = demands[:, None] + demands[None, :]
    fits = pair_sum <= capacity
    
    # Waste: remaining space after placing i then j
    waste = capacity - pair_sum
    
    # For each item j, count how many items k can be placed in the same bin as j
    # (i.e., demands[j] + demands[k] <= capacity)
    compat_count = (demands[None, :] + demands[:, None] <= capacity).sum(axis=1)
    
    # Scarcity: items with fewer compatible partners are more scarce
    # We want to place scarce items first (or pair them with compatible items)
    # For the heuristic matrix H[i, j]: attractiveness of placing j after i
    # We want to encourage placing j after i if:
    # 1. They fit (pair_sum <= capacity)
    # 2. The waste is small (they fit together well)
    # 3. Item j is scarce (has few compatible partners), so we should place it when we can
    
    # Base heuristic: prefer pairs with less waste
    # waste[i, j] is the remaining space; smaller waste means better fit
    # We want larger values for smaller waste, so use 1/(waste + epsilon) or (capacity - waste)/capacity
    
    # Normalize waste: waste ranges from 0 to capacity - min_pair
    # Use inverse waste as attractiveness, but handle zero waste carefully
    eps = 1e-9
    inv_waste = 1.0 / (waste + eps)
    
    # Scarcity bonus: multiply by scarcity of item j
    # Items with fewer compatible partners should get higher priority
    # Normalize scarcity: higher is rarer
    max_compat = compat_count.max()
    if max_compat > 0:
        scarcity = 1.0 / (compat_count + eps)
    else:
        scarcity = np.ones(n)
    
    # Combine: inv_waste * scarcity
    # This encourages placing scarce items in tight fits
    heuristic = inv_waste * scarcity[None, :]
    
    # Where they don't fit, set to 1e-9
    heuristic = np.where(fits, heuristic, 1e-9)
    
    # Ensure positive values
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
