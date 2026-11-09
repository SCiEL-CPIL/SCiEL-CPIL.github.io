"""
Guided local search penalty for TSP using a calibrated additive penalty model.

This program combines the direct additive penalty structure from high-performing 
inspirations with adaptive scaling based on problem statistics. It avoids 
expensive internal 2-opt simulations, relying instead on a robust penalty 
matrix that effectively biases the evaluator's 2-opt away from local optima.
"""
import numpy as np
import time

_DEADLINE = 330.0
_start_time = None


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Strategy:
    - Penalize edges in the current local-optimal tour strongly.
    - Add a smooth, saturating penalty for frequently used edges (diversification).
    - Add a topological bias based on node degrees in the usage graph.
    - Scale penalties relative to the typical edge distance in the instance.
    
    Returns a finite distance matrix of the same shape.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Check if we're close to the deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5:
        # Return a simple, safe penalty to be safe
        return edge_distance + 0.1 * edge_n_used / (edge_n_used.max() + 1e-10)
    
    n = edge_distance.shape[0]
    tour = list(local_opt_tour)
    
    if n < 4:
        return edge_distance.copy()
    
    # Compute reference scale: median of off-diagonal distances
    # This helps normalize penalties across different instances
    off_diag_mask = ~np.eye(n, dtype=bool)
    off_diag = edge_distance[off_diag_mask]
    scale = np.median(off_diag) if len(off_diag) > 0 else 1.0
    if scale < 1e-10:
        scale = 1.0
    
    # Build a penalty matrix
    penalty = np.zeros_like(edge_distance, dtype=np.float64)
    
    # 1) Strong penalty on edges in the current local-optimal tour
    # This is the core of Guided Local Search: force the search to break the current tour
    tour_weight = 0.35
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        penalty[u, v] += tour_weight
        penalty[v, u] += tour_weight
    
    # 2) Usage-based penalty: smooth saturating function of edge_n_used
    # Use tanh to keep penalty bounded and smooth, preventing extreme distortions
    max_usage = edge_n_used.max()
    if max_usage > 0:
        normalized_usage = edge_n_used / max_usage
        usage_penalty = np.tanh(normalized_usage)  # in [0, 1]
    else:
        usage_penalty = np.zeros_like(edge_n_used)
    
    usage_weight = 0.15
    
    # Scale penalties relative to mean distance for stability
    scaled_tour_weight = tour_weight * scale
    scaled_usage_weight = usage_weight * scale
    
    total_penalty = scaled_tour_weight * penalty + scaled_usage_weight * usage_penalty
    
    # 3) Topological bias: penalize edges incident to high-degree nodes in the usage graph
    # High-degree nodes in the usage graph are "hot spots" of previous tours.
    # Penalizing edges connecting to them encourages the tour to use less-explored nodes.
    degree = edge_n_used.sum(axis=1)
    max_degree = degree.max()
    if max_degree > 0:
        normalized_degree = degree / max_degree
        # Add a small penalty to edges incident to high-degree nodes
        degree_penalty_matrix = (normalized_degree[:, None] + normalized_degree[None, :]) * 0.5
        degree_weight = scaled_usage_weight * 0.3
        total_penalty += degree_weight * degree_penalty_matrix
    
    # 4) Adaptive refinement: Increase penalty if the current tour is "stuck"
    # Estimate how stuck we are by looking at the usage of the current tour edges
    tour_edge_usages = []
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        tour_edge_usages.append(edge_n_used[u, v])
    
    if tour_edge_usages:
        mean_tour_usage = np.mean(tour_edge_usages)
        # If the current tour edges have been used a lot, we need a stronger kick
        if mean_tour_usage > 0:
            # Normalize by max usage to get a relative measure
            relative_stuckness = mean_tour_usage / (max_usage + 1e-10)
            # Boost the tour penalty slightly if we are very stuck
            boost_factor = 1.0 + 0.5 * relative_stuckness
            total_penalty *= boost_factor
    
    # Final penalized distance matrix
    penalized = edge_distance + total_penalty
    
    # Ensure finiteness and symmetry
    # Replace any non-finite values with a large finite number
    large_val = np.max(edge_distance) * 10 + 1.0
    penalized = np.where(np.isfinite(penalized), penalized, large_val)
    
    # Enforce symmetry
    penalized = (penalized + penalized.T) / 2.0
    
    # Zero out diagonal (self-loops should not be penalized)
    np.fill_diagonal(penalized, 0.0)
    
    return penalized
