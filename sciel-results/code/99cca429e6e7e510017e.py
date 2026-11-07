"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    demands = np.asarray(demands, dtype=float)
    n = demands.shape[0]
    
    # Compatibility matrix: fits[i, j] is True if item j can follow item i
    pair_sum = demands[:, None] + demands[None, :]
    fits = pair_sum <= capacity
    
    # Waste matrix: waste[i, j] is the remaining space after placing i then j
    waste = capacity - pair_sum
    
    # Calculate scarcity for each item j:
    # Number of items that can be placed in the same bin as j
    compat_count = (demands[None, :] + demands[:, None] <= capacity).sum(axis=1)
    
    # Avoid division by zero
    scarcity = 1.0 / np.maximum(compat_count, 1)
    
    # Sharper fill score: strongly prefer tight fits using power 1.5
    fill_score = 1.0 / (waste + 1.0) ** 1.5
    
    # Forward flexibility: after placing i and j, how many items k can still fit?
    # remaining[i,j] = capacity - d[i] - d[j] = waste[i,j]
    # forward_flex[i,j] = count of k where d[k] <= waste[i,j]
    sorted_demands = np.sort(demands)
    # For each pair (i,j), waste[i,j] = capacity - d[i] - d[j]
    # We need count of demands <= waste[i,j] for each (i,j)
    # Vectorized: for each j, rem[i] = capacity - d[j] - d[i]
    forward_flex = np.zeros((n, n))
    for j in range(n):
        rem = capacity - demands[j] - demands  # shape (n,)
        forward_flex[:, j] = np.searchsorted(sorted_demands, rem, side='right')
    
    # Normalize forward flexibility to [0, 1]
    forward_flex_norm = forward_flex / max(n, 1)
    
    # Combine: fill_score * scarcity[j] * (0.5 + 0.5 * forward_flex_norm)
    # Forward flex rewards placements that leave room for more items
    heuristic = fill_score * scarcity[None, :] * (0.5 + 0.5 * forward_flex_norm)
    
    # Apply mask: if not fits, set to 1e-9
    heuristic = np.where(fits, heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
