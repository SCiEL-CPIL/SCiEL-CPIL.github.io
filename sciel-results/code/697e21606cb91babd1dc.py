"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Pair sum matrix: pair_sum[i, j] = demands[i] + demands[j]
    pair_sum = demands[:, None] + demands[None, :]
    
    # Compatibility: fits[i, j] is True if item j can be placed after item i
    fits = pair_sum <= capacity
    
    # Waste: remaining space after placing i then j
    waste = capacity - pair_sum
    
    # For each item j, compute the number of items that can be placed with it
    # (i.e., count of i such that demands[i] + demands[j] <= capacity)
    # This is the degree of j in the compatibility graph
    compat_count = fits.sum(axis=0)  # For each j, count of i that fit with j
    
    # Scarcity: inverse of compat_count, but handle zero case
    # Items with fewer compatible partners are more "scarce" and should be prioritized
    scarcity = 1.0 / (compat_count + 1e-6)
    
    # Build the heuristic matrix:
    # heuristic[i, j] should be large when:
    # 1. j fits after i (waste >= 0)
    # 2. j is scarce (few valid partners)
    # 3. The waste is small (tight fit is better for bin packing)
    
    # Base heuristic: combine scarcity of j with tightness of fit
    # For fitting pairs, use a combination of:
    #   - scarcity[j] (higher is better for scarce items)
    #   - 1/(waste+1) or similar (tighter fit is better)
    
    # Let's use: heuristic[i,j] = scarcity[j] * (1 + 1/(waste[i,j] + 1)) for fitting pairs
    # For non-fitting pairs, use 1e-9
    
    # Actually, let's think about this more carefully.
    # The heuristic guides which item to place next.
    # We want to place items that are:
    #   - Scarce (few options for where they can go)
    #   - Create tight fits (minimize waste)
    
    # heuristic[i, j] = scarcity[j] * tightness[i, j]
    # where tightness[i, j] = 1/(waste[i,j] + epsilon) for fitting pairs
    
    eps = 1e-6
    tightness = np.where(fits, 1.0 / (waste + eps), 0.0)
    
    # Combine: for each j, scale by scarcity[j]
    heuristic = tightness * scarcity[None, :]
    
    # Ensure non-positive values become 1e-9
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
