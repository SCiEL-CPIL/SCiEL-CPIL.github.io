"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step of guided local search for TSP.
    
    We penalize edges that have been frequently used in previous local optima,
    with a penalty that grows with usage. We also add a penalty to edges in the
    current tour to encourage escape from the current local optimum.
    """
    n = edge_distance.shape[0]
    
    # Build penalty from cumulative usage
    # Use a smooth, saturating function to prevent extreme penalties
    max_used = edge_n_used.max() if edge_n_used.size > 0 else 0
    
    if max_used > 0:
        # Normalize usage counts to [0, 1] with saturation
        # Use a logistic-like function for smooth saturation
        penalty_usage = 1.0 - np.exp(-0.5 * edge_n_used / max_used)
    else:
        penalty_usage = np.zeros_like(edge_n_used)
    
    # Add penalty for edges in the current tour
    tour_penalty = np.zeros_like(edge_distance)
    for i in range(n):
        j = local_opt_tour[i]
        tour_penalty[i, j] = 1.0
        tour_penalty[j, i] = 1.0
    
    # Combine penalties: usage-based penalty (scaled) + tour penalty
    # The usage penalty is scaled to be comparable to the tour penalty
    combined_penalty = 0.3 * penalty_usage + 0.2 * tour_penalty
    
    # Return modified distance matrix
    return edge_distance * (1.0 + combined_penalty)
# EVOLVE-BLOCK-END
