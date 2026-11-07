"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step of guided local search for TSP.
    
    Penalizes edges based on their cumulative usage in previous local optima.
    Uses a smooth, saturating penalty to encourage escape from the current
    local optimum while keeping the perturbed matrix well-behaved for 2-opt.
    """
    n = edge_distance.shape[0]
    
    # Start with a copy of the true distances
    perturbed = edge_distance.copy()
    
    # Get the edges in the current tour
    tour_edges = set()
    for i in range(n):
        j = (i + 1) % n
        a, b = min(i, j), max(i, j)
        tour_edges.add((a, b))
    
    # Build a penalty matrix based on edge usage
    # Use a smooth saturating function: penalty = alpha * usage / (beta + usage)
    # This grows with usage but saturates, preventing extreme distortions
    alpha = 0.6
    beta = 1.5
    
    # Compute penalty for each edge based on cumulative usage
    # edge_n_used is symmetric, so we work with the full matrix
    penalty = alpha * edge_n_used / (beta + edge_n_used)
    
    # Geometry-aware repulsive force: short edges (high value) get disproportionately
    # large penalties to force exploration of alternative short connections
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_dist = 1.0 / (edge_distance + 1e-6)
    off_diag = ~np.eye(n, dtype=bool)
    min_inv = inv_dist[off_diag].min()
    max_inv = inv_dist[off_diag].max()
    if max_inv > min_inv:
        inv_dist_norm = (inv_dist - min_inv) / (max_inv - min_inv)
    else:
        inv_dist_norm = np.zeros_like(inv_dist)
    np.fill_diagonal(inv_dist_norm, 0.0)
    
    # Scale geometry penalty to be comparable to usage penalty
    gamma = 0.4
    penalty += gamma * inv_dist_norm
    
    # Add extra penalty to edges in the current tour for immediate escape
    # This helps break the current local optimum more aggressively
    for (a, b) in tour_edges:
        penalty[a, b] += 0.4
        penalty[b, a] += 0.4
    
    # Apply the penalty multiplicatively to avoid negative distances
    perturbed = edge_distance * (1.0 + penalty)
    
    # Ensure the matrix is symmetric
    perturbed = (perturbed + perturbed.T) / 2.0
    
    # Ensure diagonal is zero
    np.fill_diagonal(perturbed, 0.0)
    
    return perturbed
# EVOLVE-BLOCK-END
