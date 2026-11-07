"""Guided local search penalty rule for TSP - multiplicative tour-edge penalty."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Args:
        edge_distance: true Euclidean distance matrix (n x n)
        local_opt_tour: current tour as a sequence of node indices
        edge_n_used: symmetric count matrix of how often each edge has been used
    
    Returns:
        Perturbed distance matrix with high penalty on current tour edges.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    n = edge_distance.shape[0]
    result = edge_distance.astype(np.float64).copy()
    
    # Get edges in the current local optimum tour
    tour_edges = []
    for i in range(len(local_opt_tour)):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % len(local_opt_tour)]
        tour_edges.append((u, v))
    
    # Apply high multiplicative penalty ONLY to edges in the current tour
    # This forces 2-opt to break these specific connections
    # while preserving the true geometry of all other edges
    
    # Use a strong but finite penalty factor
    # The penalty should be high enough to make tour edges unattractive
    # but not so high that it creates numerical issues or extreme distortion
    
    # Scale penalty slightly with usage: edges used more often get slightly more penalty
    max_used = edge_n_used.max()
    if max_used > 0:
        # Gentle scaling: base penalty 5.0, up to 10.0 for most-used edges
        base_penalty = 5.0
        usage_boost = 5.0 * (edge_n_used / (1.0 + max_used))
    else:
        base_penalty = 5.0
        usage_boost = np.zeros_like(edge_n_used)
    
    # Apply penalty to tour edges only
    for u, v in tour_edges:
        penalty = base_penalty + usage_boost[u, v]
        result[u, v] = result[u, v] * penalty
        result[v, u] = result[v, u] * penalty
    
    # Ensure symmetry (should already be symmetric, but just in case)
    result = (result + result.T) / 2.0
    
    # Ensure diagonal is zero
    np.fill_diagonal(result, 0.0)
    
    # Ensure all values are finite and positive
    result = np.where(np.isfinite(result) & (result > 0), result, edge_distance)
    np.fill_diagonal(result, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        # Return the best result we have
        pass
    
    return result
# EVOLVE-BLOCK-END
