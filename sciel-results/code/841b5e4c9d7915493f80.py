"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step of guided local search for TSP.
    
    We penalize edges that have been frequently used in previous local optima,
    with a penalty that grows with usage but saturates to avoid over-penalizing.
    We also add a small immediate penalty to edges in the current tour to encourage
    escape from the current local optimum.
    """
    n = edge_distance.shape[0]
    
    # Build penalty from cumulative usage with saturating normalization
    # Use a logistic-like saturation to prevent extreme penalties
    max_used = edge_n_used.max()
    if max_used > 0:
        # Saturating penalty: grows with usage but caps out
        # alpha controls the strength of the penalty
        alpha = 0.5
        penalty = alpha * (1.0 - np.exp(-edge_n_used / (1.0 + max_used / 10.0)))
    else:
        penalty = np.zeros_like(edge_distance)
    
    # Add immediate penalty to edges in the current local optimum tour
    # This encourages escaping the current local optimum
    tour_edges = np.zeros_like(edge_distance)
    for i in range(len(local_opt_tour) - 1):
        u, v = local_opt_tour[i], local_opt_tour[i + 1]
        tour_edges[u, v] = 1.0
        tour_edges[v, u] = 1.0
    if len(local_opt_tour) > 1:
        u, v = local_opt_tour[-1], local_opt_tour[0]
        tour_edges[u, v] = 1.0
        tour_edges[v, u] = 1.0
    
    # Immediate penalty weight
    beta = 0.3
    penalty += beta * tour_edges
    
    # Apply penalty to edge distances
    return edge_distance + penalty
# EVOLVE-BLOCK-END
