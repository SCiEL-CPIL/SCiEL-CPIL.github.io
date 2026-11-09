"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    GLS penalty: penalize edges in the current local optimum tour
    based on their historical usage frequency.
    """
    n = len(local_opt_tour)
    if n <= 1:
        return edge_distance.copy()
    
    result = edge_distance.copy()
    
    # Scale penalty relative to typical edge distances
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    med_dist = np.median(edge_distance[mask])
    if med_dist <= 0:
        med_dist = 1.0
    
    # Penalty per unit of usage count
    penalty_coeff = 0.15 * med_dist
    
    # Only penalize edges that are in the current local optimum tour
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        if u == v:
            continue
        count = edge_n_used[u, v]
        if count > 0:
            penalty = penalty_coeff * count
            result[u, v] += penalty
            result[v, u] += penalty
    
    return result
# EVOLVE-BLOCK-END
