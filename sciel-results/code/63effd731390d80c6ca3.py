"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_GLOBAL_START_TIME = time.time()
_WALL_CLOCK_LIMIT = 325.0  # Safety margin before 330s kill

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # Check time limit - return original distances if we're running out of time
    if time.time() - _GLOBAL_START_TIME > _WALL_CLOCK_LIMIT:
        return edge_distance.copy()
    
    # Create a copy of the true distance matrix to modify
    modified_dist = edge_distance.copy()
    
    n = edge_distance.shape[0]
    
    # Identify edges in the current local optimum tour
    # local_opt_tour is a permutation of [0, n-1]
    # Edges are (tour[i], tour[i+1]) for i in 0..n-2 and (tour[n-1], tour[0])
    
    # Strategy: Hard Exclusion
    # Set the distance of all edges in the current tour to a very large value.
    # This forces the 2-opt operator to remove these edges, ensuring a significant
    # change in the tour structure rather than a minor perturbation.
    
    # Create indices for the tour edges
    i_idx = np.arange(n)
    j_idx = (i_idx + 1) % n
    
    u = local_opt_tour[i_idx]
    v = local_opt_tour[j_idx]
    
    # Set the distance to a large finite value.
    # We use a value much larger than the maximum possible tour length (100 * sqrt(2) ~ 141).
    # 1000.0 is safe and effectively infinite for 2-opt comparisons within the matrix.
    LARGE_VALUE = 1000.0
    
    modified_dist[u, v] = LARGE_VALUE
    modified_dist[v, u] = LARGE_VALUE
    
    return modified_dist
# EVOLVE-BLOCK-END
