"""Guided local search penalty rule for the traveling salesman problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for guided local search on TSP.
    
    Applies penalties to edges based on their usage frequency, with emphasis
    on edges in the current local optimum tour. Uses a smooth, adaptive penalty
    that scales with both usage count and edge length to encourage exploration
    of alternative routes.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    n = edge_distance.shape[0]
    
    # Get edges in the current local optimum tour
    tour = np.array(local_opt_tour)
    if len(tour) < 2:
        return edge_distance.copy()
    
    # Extract tour edges (including the closing edge)
    tour_edges = []
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        tour_edges.append((u, v))
    
    # Build a set of tour edges for fast lookup
    tour_edge_set = set()
    for u, v in tour_edges:
        tour_edge_set.add((min(u, v), max(u, v)))
    
    # Compute usage statistics
    max_used = edge_n_used.max()
    if max_used > 0:
        # Normalize usage to [0, 1]
        normalized_usage = edge_n_used / max_used
    else:
        normalized_usage = np.zeros_like(edge_n_used)
    
    # Create penalty matrix
    penalty = np.zeros_like(edge_distance)
    
    # Identify edges in the current tour and apply stronger penalties
    for u, v in tour_edges:
        usage = edge_n_used[u, v]
        # Penalty is proportional to usage and inversely proportional to edge length
        # This encourages replacing short, frequently-used edges
        d = edge_distance[u, v]
        if d > 1e-10:
            # Use a smooth penalty that grows with usage
            # The factor 0.5 gives moderate penalty strength
            penalty[u, v] = 0.5 * usage / (1.0 + usage) / (d + 0.1)
            penalty[v, u] = penalty[u, v]
    
    # Apply lighter penalties to non-tour edges that are frequently used
    # This helps avoid reusing edges that were in previous local optima
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in tour_edge_set:
                usage = edge_n_used[u, v]
                if usage > 0:
                    d = edge_distance[u, v]
                    if d > 1e-10:
                        # Lighter penalty for non-tour edges
                        penalty[u, v] = 0.2 * usage / (1.0 + usage) / (d + 0.1)
                        penalty[v, u] = penalty[u, v]
    
    # Apply the penalty to the distance matrix
    # Use an exponential form for smoothness
    perturbed = edge_distance * (1.0 + penalty)
    
    # Ensure the result is finite and has the same shape
    perturbed = np.where(np.isfinite(perturbed), perturbed, edge_distance)
    
    # Check deadline
    if time.time() > deadline:
        return perturbed
    
    return perturbed
# EVOLVE-BLOCK-END
