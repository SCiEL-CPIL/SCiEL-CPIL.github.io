"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    # Effective cost of visiting node j from i: d[i,j] + d[j,0]
    # This is the additional path length incurred
    return_cost = distance_matrix + distance_matrix[:, 0:1]  # d[i,j] + d[j,0]
    
    # Clip to avoid division by zero
    return_cost = np.maximum(return_cost, 1e-9)
    
    # Prize per unit effective cost, with power law to emphasize distance
    # alpha=1.5 gives stronger distance penalty than simple ratio
    alpha = 1.5
    heuristic = prizes[None, :] / (return_cost ** alpha)
    
    # Ensure positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Normalize to keep values in a reasonable range
    # Scale so that max is around 10 (gives good dynamic range for pheromone*heuristic)
    max_val = np.max(heuristic)
    if max_val > 0:
        heuristic = heuristic * (10.0 / max_val)
    
    return heuristic
# EVOLVE-BLOCK-END
