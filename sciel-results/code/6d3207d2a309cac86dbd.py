"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Applies a penalty to edges that have been frequently used in previous tours.
    The penalty is scaled by a non-linear function of the edge usage count,
    weighted inversely by edge length to create stronger pressure on short edges.
    
    Args:
        edge_distance: true Euclidean distance matrix (n x n)
        local_opt_tour: current tour as a sequence of node indices
        edge_n_used: symmetric count matrix of how often each edge has been used
    
    Returns:
        Perturbed distance matrix with penalties applied to frequently used edges.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # 300s limit, 10s safety margin
    
    n = edge_distance.shape[0]
    
    # Create a copy to avoid modifying the original
    perturbed = edge_distance.copy()
    
    # Get the edges in the current tour
    tour = list(local_opt_tour)
    if len(tour) < 2:
        return perturbed
    
    # Ensure the tour is closed (last node connects back to first)
    tour_edges = []
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        tour_edges.append((u, v))
    
    # Normalize edge usage counts
    max_used = edge_n_used.max()
    if max_used > 0:
        # Non-linear scaling: use sqrt to amplify differences
        # Edges with higher usage get disproportionately larger penalties
        usage_factor = np.sqrt(edge_n_used) / np.sqrt(max_used)
    else:
        usage_factor = np.zeros_like(edge_n_used)
    
    # Compute penalty for each edge in the tour
    # Penalty is inversely proportional to edge length (shorter edges get more penalty)
    # This forces exploration of alternative short connections
    
    # Base penalty rate - tuned for 100-city TSP
    base_penalty = 0.15
    
    for u, v in tour_edges:
        if u == v:
            continue
        
        # Get the true distance for this edge
        d = edge_distance[u, v]
        
        if d <= 0:
            continue
        
        # Inverse distance weighting: shorter edges get larger relative penalty
        # Use a softened inverse to avoid division by zero issues
        inv_dist_weight = 1.0 / (d + 0.01)
        
        # Combine usage factor and inverse distance weight
        # The penalty grows with usage and is stronger for shorter edges
        penalty = base_penalty * usage_factor[u, v] * inv_dist_weight
        
        # Apply penalty to both symmetric entries
        perturbed[u, v] += penalty
        perturbed[v, u] += penalty
    
    # Check deadline - if we're running out of time, return immediately
    if time.time() > deadline - 5.0:
        # Ensure the matrix is valid (finite, same shape)
        perturbed = np.where(np.isfinite(perturbed), perturbed, edge_distance)
        return perturbed
    
    # Optional: apply a global scaling to ensure penalties are significant
    # but not so large that they dominate the true distances
    # This helps 2-opt find meaningful improvements
    
    # Verify the result is finite and has the correct shape
    if not np.all(np.isfinite(perturbed)):
        perturbed = np.where(np.isfinite(perturbed), perturbed, edge_distance)
    
    if perturbed.shape != edge_distance.shape:
        perturbed = edge_distance.copy()
    
    return perturbed
# EVOLVE-BLOCK-END
