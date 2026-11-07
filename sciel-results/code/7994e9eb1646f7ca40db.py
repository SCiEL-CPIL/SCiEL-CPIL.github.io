"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty rule for TSP.
    
    Implements a strong edge-exclusion strategy: edges in the current local-optimal
    tour are penalized heavily (effectively excluded) to force the subsequent 2-opt
    to remove them, guaranteeing a distinct neighborhood search step.
    
    Args:
        edge_distance: True Euclidean distance matrix (n x n).
        local_opt_tour: Current tour as a list/array of node indices (closed tour).
        edge_n_used: Symmetric count matrix of how many times each edge has been used.
    
    Returns:
        A modified distance matrix of the same shape with finite values.
    """
    n = edge_distance.shape[0]
    modified = edge_distance.copy().astype(float)
    
    # Ensure finite values
    modified = np.where(np.isfinite(modified), modified, 0.0)
    
    # Extract edges from the current tour
    # The tour is a closed tour: tour[0] -> tour[1] -> ... -> tour[-1] -> tour[0]
    tour = np.asarray(local_opt_tour)
    
    # Handle potential edge cases
    if len(tour) < 2:
        return modified
    
    # Get the set of edges in the current tour
    tour_edges = set()
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        if u != v:
            tour_edges.add((min(u, v), max(u, v)))
    
    # Penalty strategy:
    # 1. Heavily penalize all edges in the current tour to force their removal
    # 2. Also apply a softer penalty based on edge_n_used for additional diversity
    
    # Base penalty for tour edges - make them very unattractive
    # Use a large but finite value to avoid issues with inf
    max_dist = np.max(edge_distance[np.isfinite(edge_distance)]) if np.any(np.isfinite(edge_distance)) else 1.0
    if max_dist <= 0:
        max_dist = 1.0
    
    # Penalty multiplier for tour edges: make them ~10x the max distance
    tour_penalty = 10.0 * max_dist
    
    # Apply tour edge penalties
    for (u, v) in tour_edges:
        modified[u, v] += tour_penalty
        modified[v, u] += tour_penalty
    
    # Additional soft penalty based on edge_n_used for non-tour edges
    # This adds diversity to the search by penalizing frequently used edges
    if edge_n_used is not None and edge_n_used.size > 0:
        n_used = np.asarray(edge_n_used)
        # Normalize edge_n_used to [0, 1]
        max_used = np.max(n_used)
        if max_used > 0:
            normalized_used = n_used / max_used
            # Apply a smaller penalty to frequently used edges (not in current tour)
            # Only apply to edges not already in the tour to avoid double-counting
            soft_penalty = 2.0 * max_dist * normalized_used
            # Don't apply soft penalty to tour edges (they already have heavy penalty)
            for (u, v) in tour_edges:
                soft_penalty[u, v] = 0.0
                soft_penalty[v, u] = 0.0
            modified += soft_penalty
    
    # Ensure all values are finite
    modified = np.where(np.isfinite(modified), modified, 0.0)
    
    # Ensure symmetry
    modified = (modified + modified.T) / 2.0
    
    return modified
# EVOLVE-BLOCK-END
