"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty rule for TSP.
    
    Combines two mechanisms:
    1. Historical penalty: penalize edges based on accumulated usage count
    2. Tour-aware boost: strongly penalize edges in the current local optimum
       to force escape from the current basin
    
    Uses exponential saturation for smooth, bounded penalty behavior.
    """
    n = len(local_opt_tour)
    if n <= 2:
        return edge_distance.copy()
    
    # Check deadline
    if time.time() > _DEADLINE:
        return edge_distance.copy()
    
    # Create a copy of the true distance matrix
    perturbed_dist = edge_distance.copy()
    
    # Historical penalty: use exponential saturation to avoid over-penalization
    # Penalty grows with usage but saturates smoothly
    max_used = edge_n_used.max()
    if max_used > 0:
        # Normalized usage in [0, 1]
        normalized_used = edge_n_used / (1.0 + max_used)
        # Exponential saturation: penalty factor in [1, 1+alpha]
        alpha_hist = 0.3
        hist_penalty = 1.0 + alpha_hist * (1.0 - np.exp(-2.0 * normalized_used))
        perturbed_dist *= hist_penalty
    
    # Tour-aware boost: strongly penalize edges in the current local optimum
    # This directly breaks the current solution structure
    tour_edges = np.zeros_like(edge_distance)
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_edges[u, v] = 1.0
        tour_edges[v, u] = 1.0
    
    # Adaptive boost: stronger when the tour has been used more
    # Average usage of tour edges
    tour_avg_usage = np.mean(edge_n_used[tour_edges == 1.0])
    
    # Boost factor: base + adaptive component
    base_boost = 1.5
    adaptive_boost = 0.5 * (1.0 - np.exp(-0.5 * tour_avg_usage))
    total_boost = base_boost + adaptive_boost
    
    # Apply boost to tour edges
    perturbed_dist[tour_edges == 1.0] *= total_boost
    
    # Ensure finite values
    perturbed_dist = np.nan_to_num(perturbed_dist, nan=0.0, posinf=edge_distance.max() * 10, neginf=0.0)
    
    return perturbed_dist
# EVOLVE-BLOCK-END
