"""Guided local search penalty rule for TSP - topological disruption strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search using topological disruption.
    
    Instead of just adding scalar penalties, we identify the most 'critical' edges
    in the current tour (shortest edges that form local structures) and disrupt them
    by setting their cost to a large value, forcing the 2-opt to restructure.
    
    Args:
        edge_distance: (N, N) array of true Euclidean distances.
        local_opt_tour: (N,) array of node indices forming the current tour.
        edge_n_used: (N, N) array of symmetric counts of edge usage.
        
    Returns:
        (N, N) array of perturbed distances.
    """
    N = edge_distance.shape[0]
    
    # Start with a copy of the true distances
    perturbed = edge_distance.copy()
    
    # Extract tour edges
    tour = local_opt_tour
    tour_edges = []
    for i in range(N):
        u = tour[i]
        v = tour[(i + 1) % N]
        tour_edges.append((u, v))
    
    # Compute edge lengths for tour edges
    edge_lengths = np.array([edge_distance[u, v] for u, v in tour_edges])
    
    # Strategy: identify the k shortest edges in the tour and disrupt them
    # These are the "structural" edges that 2-opt tends to preserve
    # We use a dynamic k based on the tour length
    k = max(1, N // 10)  # Disrupt about 10% of the shortest edges
    
    # Get indices of the k shortest edges
    sorted_indices = np.argsort(edge_lengths)
    critical_edge_indices = sorted_indices[:k]
    
    # Also consider edges that have been used many times (high frequency in edge_n_used)
    # Combine: edges that are both short AND frequently used are most critical
    usage_counts = np.array([edge_n_used[u, v] for u, v in tour_edges])
    
    # Score each edge: combination of shortness and usage
    # Normalize lengths and usage to [0, 1]
    if edge_lengths.max() > edge_lengths.min():
        norm_lengths = (edge_lengths - edge_lengths.min()) / (edge_lengths.max() - edge_lengths.min())
    else:
        norm_lengths = np.zeros(N)
    
    if usage_counts.max() > usage_counts.min():
        norm_usage = (usage_counts - usage_counts.min()) / (usage_counts.max() - usage_counts.min())
    else:
        norm_usage = np.zeros(N)
    
    # Critical score: short edges with high usage are most critical
    critical_score = 0.6 * (1.0 - norm_lengths) + 0.4 * norm_usage
    
    # Select top k most critical edges
    critical_indices = np.argsort(critical_score)[-k:]
    
    # Set these edges to a large penalty value
    # Use a value that's large enough to force restructuring but not infinite
    # (to avoid numerical issues)
    max_dist = edge_distance.max()
    penalty_value = max_dist * 10.0
    
    for idx in critical_indices:
        u, v = tour_edges[idx]
        perturbed[u, v] = penalty_value
        perturbed[v, u] = penalty_value
    
    # Additionally, add a mild penalty to all tour edges based on usage
    # This provides a smooth gradient for the search
    if edge_n_used.max() > 0:
        usage_norm = edge_n_used / (1.0 + edge_n_used.max())
        # Only apply to tour edges to avoid over-penalizing
        for u, v in tour_edges:
            if perturbed[u, v] < penalty_value:  # Skip already disrupted edges
                perturbed[u, v] *= (1.0 + 0.05 * usage_norm[u, v])
                perturbed[v, u] *= (1.0 + 0.05 * usage_norm[v, u])
    
    # Ensure symmetry
    perturbed = (perturbed + perturbed.T) / 2.0
    
    # Ensure diagonal is 0
    np.fill_diagonal(perturbed, 0.0)
    
    # Ensure all values are finite
    perturbed = np.where(np.isfinite(perturbed), perturbed, max_dist)
    
    return perturbed
# EVOLVE-BLOCK-END
