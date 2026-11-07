"""Guided local search penalty for TSP with 100 points."""
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Compute a modified distance matrix for guided local search.
    
    Parameters:
    - edge_distance: (n, n) true Euclidean distance matrix
    - local_opt_tour: list/array of city indices forming the current tour
    - edge_n_used: (n, n) symmetric count matrix of edge usage
    
    Returns:
    - Modified distance matrix with penalties added
    """
    n = edge_distance.shape[0]
    
    # Get current tour edges
    tour_edges = set()
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_edges.add((min(u, v), max(u, v)))
    
    # Compute adaptive penalty scale based on median edge distance
    # Use non-diagonal entries
    mask = ~np.eye(n, dtype=bool)
    nonzero_mask = (edge_distance > 1e-10) & mask
    if np.any(nonzero_mask):
        dist_scale = np.median(edge_distance[nonzero_mask])
    else:
        dist_scale = 1.0
    
    # Normalize usage counts
    max_used = edge_n_used.max() if edge_n_used.max() > 0 else 1.0
    normalized_usage = edge_n_used / max_used
    
    # Base penalty: proportional to usage, scaled by median distance
    # Use a moderate coefficient to encourage exploration without being too aggressive
    penalty_coeff = 0.3 * dist_scale
    
    # Apply penalty only to edges in the current local optimum tour
    # This is the core of guided local search: penalize edges of the current local optimum
    penalty_matrix = np.zeros_like(edge_distance)
    for (u, v) in tour_edges:
        penalty_matrix[u, v] = penalty_coeff * normalized_usage[u, v]
        penalty_matrix[v, u] = penalty_matrix[u, v]
    
    # Add a small base penalty to all tour edges to ensure they are always discouraged
    # even if their usage count is low
    base_penalty = 0.05 * dist_scale
    for (u, v) in tour_edges:
        penalty_matrix[u, v] += base_penalty
        penalty_matrix[v, u] += base_penalty
    
    # Return modified distance matrix
    return edge_distance + penalty_matrix
