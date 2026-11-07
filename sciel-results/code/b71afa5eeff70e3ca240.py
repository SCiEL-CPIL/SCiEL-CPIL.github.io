import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    start_time = time.time()
    n = edge_distance.shape[0]
    # Copy the matrix to avoid side effects
    perturbed = edge_distance.copy()
    
    # Get the edges of the current tour
    tour_edges = []
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_edges.append((u, v))
        
    # Calculate mean edge length in the current tour
    if len(tour_edges) > 0:
        tour_lengths = [edge_distance[u, v] for u, v in tour_edges]
        mean_edge_len = np.mean(tour_lengths)
    else:
        mean_edge_len = 1.0 # Fallback, shouldn't happen for n>=2
        
    # Penalty parameter: use non-linear (power 1.5) scaling for stronger selective perturbation
    # This creates stronger penalties for persistent edges, enabling better escape from local optima
    penalty_scale = 0.15 * mean_edge_len
    
    # Apply non-linear penalty to edges in the current tour
    for u, v in tour_edges:
        count = edge_n_used[u, v]
        # If count is 0, no penalty.
        # Use power 1.5 scaling for non-linear, targeted perturbation
        if count > 0:
            penalty = penalty_scale * (count ** 1.5)
            perturbed[u, v] += penalty
            perturbed[v, u] += penalty
            
    # Time guard: if we're close to the deadline, return immediately
    elapsed = time.time() - start_time
    if elapsed > 280:  # Safety margin before 300s limit
        return perturbed
        
    return perturbed
