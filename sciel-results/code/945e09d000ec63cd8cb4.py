"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


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
        usage_penalty = alpha * edge_n_used / (1.0 + edge_n_used)
    else:
        usage_penalty = np.zeros_like(edge_n_used)
    
    # Add immediate penalty to edges in the current tour
    tour_penalty = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        j = local_opt_tour[i]
        k = local_opt_tour[(i + 1) % n]
        tour_penalty[i, j] += 1.0
        tour_penalty[j, i] += 1.0
    
    # Combine: scale usage penalty and add tour penalty
    # The tour penalty is small to encourage escape but not dominate
    beta = 0.3
    total_penalty = usage_penalty + beta * tour_penalty
    
    # Apply additive perturbation instead of multiplicative
    # This ensures all edges are penalized by a positive amount,
    # but the relative increase is smaller for longer edges
    # which helps preserve the structure of the distance matrix
    # while still pushing 2-opt to explore different edges
    
    # Scale the penalty to be comparable to typical edge lengths
    # For 100 points in unit square, typical edge length ~ 0.1
    # We want the penalty to be a meaningful fraction of this
    mean_dist = np.mean(edge_distance[np.triu_indices(n, k=1)])
    penalty_scale = 0.1 * mean_dist
    
    perturbed = edge_distance + penalty_scale * total_penalty
    
    # Safety: ensure finite values
    perturbed = np.where(np.isfinite(perturbed), perturbed, edge_distance)
    
    return perturbed
# EVOLVE-BLOCK-END
