"""Guided local search penalty rule for the TSP."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search.
    
    Adds penalties to edges in the current local optimal tour, scaled by 
    their usage frequency. This forces the subsequent local search to 
    explore a different region of the solution space.
    """
    # Deadline guard (though this function is O(N), it's good practice)
    t0 = time.time()
    DEADLINE = 330.0
    
    n = edge_distance.shape[0]
    
    # Calculate mean edge distance for scaling the penalty
    # We exclude the diagonal (0) for a better estimate of typical edge cost
    off_diag_mask = ~np.eye(n, dtype=bool)
    mean_dist = np.mean(edge_distance[off_diag_mask])
    
    # Start with a copy of the true distances
    penalized_distance = edge_distance.copy()
    
    # Extract edges from the tour
    # local_opt_tour is a sequence of nodes, e.g., [0, 5, 2, ...]
    # The edges are (tour[0], tour[1]), (tour[1], tour[2]), ..., (tour[-1], tour[0])
    tour = local_opt_tour
    
    # Apply penalties to each edge in the tour
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        
        # Get the usage count for this edge
        # edge_n_used is symmetric, so we can just pick one direction
        usage = edge_n_used[u, v]
        
        # Penalty magnitude:
        # Base penalty + usage-dependent increment
        # We want to make these edges significantly more expensive than average.
        # Using a factor of 1.0 * mean_dist * (1 + usage)
        # This ensures that even unused tour edges are penalized, and frequently
        # used edges are heavily penalized.
        penalty = 1.0 * mean_dist * (1.0 + usage)
        
        # Add penalty to both directions
        penalized_distance[u, v] += penalty
        penalized_distance[v, u] += penalty
        
        # Time check (extremely unlikely to trigger for N=100)
        if time.time() - t0 > DEADLINE - 10.0:
            break
            
    # Ensure the matrix is finite (it should be, but just in case)
    # np.nan_to_num handles any potential inf/nan from edge cases
    penalized_distance = np.nan_to_num(penalized_distance, nan=0.0, posinf=np.finfo(float).max/2, neginf=0.0)
    
    return penalized_distance
# EVOLVE-BLOCK-END
