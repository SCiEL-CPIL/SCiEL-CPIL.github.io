"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    n = len(local_opt_tour)
    if n == 0:
        return edge_distance.copy()
    
    # Create a copy of the distance matrix
    penalized = edge_distance.copy()
    
    # Get the edges in the current local optimal tour
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        
        # Apply penalty based on usage count
        count = edge_n_used[u, v]
        if count > 0:
            # Stronger penalty for frequently used edges
            # Use a scaling factor that grows with usage
            penalty_factor = 1.0 + 0.5 * count
            penalized[u, v] *= penalty_factor
            penalized[v, u] *= penalty_factor
    
    # Ensure all values are finite
    penalized[~np.isfinite(penalized)] = np.finfo(float).max / 1000
    
    return penalized
# EVOLVE-BLOCK-END
