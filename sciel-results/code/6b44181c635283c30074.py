"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided Local Search perturbation step.
    Adds an additive penalty to edges in the current local optimum,
    proportional to their historical usage count.
    """
    perturbed = edge_distance.copy()
    
    # Penalty coefficient
    alpha = 0.15
    
    n = len(local_opt_tour)
    if n == 0:
        return perturbed
        
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        
        # Penalty is proportional to how often this edge has been used
        # in previous local optima.
        usage = edge_n_used[u, v]
        if usage > 0:
            penalty = alpha * usage
            perturbed[u, v] += penalty
            perturbed[v, u] += penalty

    return perturbed
# EVOLVE-BLOCK-END
