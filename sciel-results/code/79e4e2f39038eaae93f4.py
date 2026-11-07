import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty for TSP.
    
    Applies an additive penalty:
    1. Linear penalty based on cumulative edge usage (edge_n_used)
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
    
    # Additive penalty parameters
    # Alpha usage: penalty per unit of usage count
    # Alpha tour: penalty for being in the current local optimum
    # Values tuned for unit square 100 points (avg edge ~0.07)
    alpha_usage = 0.05
    alpha_tour = 0.1
    
    # Apply linear penalty for usage
    # edge_n_used is symmetric
    d_out += alpha_usage * edge_n_used
    
    # Apply penalty to current tour edges
    d_out[tour_edges] += alpha_tour
    
    # Ensure diagonal stays 0 (or original value, which should be 0)
    np.fill_diagonal(d_out, 0.0)
    
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
