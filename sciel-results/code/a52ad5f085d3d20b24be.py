"""Guided local search penalty for TSP; Escalating multiplicative penalty on repeated tour edges."""
# EVOLVE-BLOCK-START
import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Applies an escalating multiplicative penalty to edges in the current local optimum 
    based on their historical usage count.
    
    Args:
    edge_distance: np.ndarray, true distance matrix.
    local_opt_tour: list or np.ndarray, the current tour as a sequence of node indices.
    edge_n_used: np.ndarray, symmetric matrix of usage counts.
    
    Returns:
    np.ndarray: Modified distance matrix for the next 2-opt phase.
    """
    n = len(local_opt_tour)
    if n < 2:
        return edge_distance.copy()
    
    # Copy the true distances
    new_dist = edge_distance.copy()
    
    # Iterate over edges in the current tour
    # The tour is a cycle: 0->1, 1->2, ..., n-1->0
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        
        # Get usage count for this edge
        count = edge_n_used[u, v]
        
        if count > 0:
            # Escalating penalty: 
            # If used 1 time, factor = 1.5
            # If used 2 times, factor = 2.0
            # If used 3 times, factor = 2.5
            # Capped at factor 4.0 (count 6) to avoid extreme distortion
            capped_count = min(count, 6.0)
            factor = 1.0 + 0.5 * capped_count
            
            new_dist[u, v] = edge_distance[u, v] * factor
            new_dist[v, u] = new_dist[u, v]
            
    return new_dist
# EVOLVE-BLOCK-END
