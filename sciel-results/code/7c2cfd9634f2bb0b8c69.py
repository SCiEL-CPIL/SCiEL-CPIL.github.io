import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty for TSP.
    
    Applies a composite penalty:
    1. Saturating penalty based on cumulative edge usage (edge_n_used)
    2. Additional immediate penalty for edges in the current local optimum tour
    
    Returns a perturbed distance matrix that guides 2-opt away from 
    frequently-used and currently-used edges.
    """
    start_time = time.time()
    n = edge_distance.shape[0]
    
    # Work on a copy to avoid modifying the input
    d_out = edge_distance.copy()
    
    if n <= 1:
        return d_out
    
    # Get edges in the current local optimum tour
    # local_opt_tour is a sequence of city indices forming a closed tour
    tour = np.array(local_opt_tour)
    if len(tour) < 2:
        return d_out
    
    # Identify edges in the current tour
    tour_edges = np.zeros((n, n), dtype=bool)
    for i in range(len(tour) - 1):
        a, b = tour[i], tour[i + 1]
        tour_edges[a, b] = True
        tour_edges[b, a] = True
    # Close the tour
    a, b = tour[-1], tour[0]
    tour_edges[a, b] = True
    tour_edges[b, a] = True
    
    # Compute saturating penalty based on cumulative usage
    # Use n / (n + k) where k controls saturation level
    # Larger k = slower saturation = more linear for small counts
    k = 5.0  # saturation parameter
    usage_penalty = edge_n_used / (edge_n_used + k)
    
    # Base penalty coefficient for usage
    alpha_usage = 0.8
    
    # Additional penalty for edges in the current tour
    alpha_tour = 0.5
    
    # Composite penalty multiplier
    # Start with 1.0 (no change), add penalties
    penalty_multiplier = 1.0 + alpha_usage * usage_penalty
    
    # Apply additional penalty to tour edges
    penalty_multiplier[tour_edges] += alpha_tour
    
    # Ensure diagonal stays 0 (or original value)
    np.fill_diagonal(penalty_multiplier, 1.0)
    
    # Apply penalty
    d_out = edge_distance * penalty_multiplier
    
    # Ensure symmetry (should already be symmetric, but enforce it)
    d_out = (d_out + d_out.T) / 2.0
    
    # Ensure finite values
    d_out = np.nan_to_num(d_out, nan=0.0, posinf=1e10, neginf=1e10)
    
    # Ensure non-negative distances
    d_out = np.maximum(d_out, 0.0)
    
    # Deadline guard: if we're running low on time, return what we have
    elapsed = time.time() - start_time
    if elapsed > 280.0:  # Safety margin before 300s limit
        return d_out
    
    return d_out
