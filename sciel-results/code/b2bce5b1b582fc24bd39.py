"""Guided local search penalty rule for TSP with binary threshold penalties."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.perf_counter()
_DEADLINE = _start_time + 325.0

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Apply a binary threshold penalty to the most frequently used edges in the current tour.
    This forces the local search to escape the specific basin of attraction by making
    the 'stuck' edges prohibitively expensive, while leaving the rest of the landscape
    undisturbed.
    """
    if time.perf_counter() > _DEADLINE:
        return edge_distance.copy()

    n = len(local_opt_tour)
    if n == 0:
        return edge_distance.copy()

    # Ensure we have a working copy to avoid modifying the input
    modified_distance = edge_distance.copy()
    
    # Extract the edges in the current tour
    # local_opt_tour is a list of node indices representing the cycle
    # Edges are (tour[i], tour[(i+1) % n])
    tour_edges = []
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_edges.append((u, v))
        
    # Get the usage counts for these specific edges
    # edge_n_used is symmetric, so (u,v) and (v,u) are the same
    usage_counts = np.array([edge_n_used[u, v] for u, v in tour_edges])
    
    if len(usage_counts) == 0:
        return modified_distance

    # Strategy: Penalize the top 20% of most-used edges in the current tour.
    # This targets the "core" of the local optimum that has resisted change.
    if len(usage_counts) >= 5:
        threshold_index = int(0.2 * len(usage_counts))
        # Find indices of the top threshold_index counts
        # We want the indices of the largest values
        top_indices = np.argsort(usage_counts)[-threshold_index:]
    else:
        # If tour is small, penalize the single most used edge if any
        if usage_counts.max() > 0:
            top_indices = np.argmax(usage_counts)
            top_indices = [top_indices]
        else:
            return modified_distance
            
    # Calculate the penalty magnitude
    # Use a significant portion of the average edge length to ensure the swap is forced
    # Average edge length in the current tour
    avg_tour_len = np.mean([edge_distance[u, v] for u, v in tour_edges])
    # Penalty should be large enough to force 2-opt to remove these edges
    # 1.5 * avg_len is a strong penalty
    penalty = 1.5 * avg_tour_len
    
    # Apply the penalty to the selected edges
    for idx in top_indices:
        u, v = tour_edges[idx]
        modified_distance[u, v] += penalty
        modified_distance[v, u] += penalty
        
    return modified_distance
# EVOLVE-BLOCK-END
