"""Rewritten guided local search penalty for TSP."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 325.0
_start_time = None


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Strategy:
    - Penalize edges in the current local-optimal tour more strongly.
    - Add a global penalty proportional to accumulated usage (edge_n_used).
    - Use a smooth, saturating penalty to avoid over-penalizing.
    - Add a small spectral/Laplacian-like smoothing term to bias
      against high-frequency tour oscillations (global topological bias).
    
    Returns a finite distance matrix of the same shape.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = edge_distance.shape[0]
    
    # Build a penalty matrix: strong penalty on current tour edges,
    # plus a usage-based penalty for all edges.
    penalty = np.zeros_like(edge_distance, dtype=np.float64)
    
    # 1) Strong penalty on edges in the current local-optimal tour
    for i in range(n):
        j = local_opt_tour[i]
        penalty[i, j] = 1.0
        penalty[j, i] = 1.0
    
    # 2) Usage-based penalty: smooth saturating function of edge_n_used
    # Use tanh to keep penalty bounded and smooth
    max_usage = edge_n_used.max()
    if max_usage > 0:
        normalized_usage = edge_n_used / max_usage
        usage_penalty = np.tanh(normalized_usage)  # in [0, 1]
    else:
        usage_penalty = np.zeros_like(edge_n_used)
    
    # Combine: tour edges get a strong base penalty, all edges get usage penalty
    # Weight the tour penalty more heavily to escape the current local optimum
    tour_weight = 0.35
    usage_weight = 0.15
    
    # Scale penalties relative to mean distance for stability
    mean_dist = np.mean(edge_distance)
    if mean_dist > 0:
        scaled_tour_weight = tour_weight * mean_dist
        scaled_usage_weight = usage_weight * mean_dist
    else:
        scaled_tour_weight = tour_weight
        scaled_usage_weight = usage_weight
    
    total_penalty = scaled_tour_weight * penalty + scaled_usage_weight * usage_penalty
    
    # 3) Optional: spectral/Laplacian-inspired smoothing bias
    # Compute node degrees from edge_n_used
    degree = edge_n_used.sum(axis=1)
    max_degree = degree.max()
    if max_degree > 0:
        normalized_degree = degree / max_degree
        # Add a small penalty to edges incident to high-degree nodes
        degree_penalty = (normalized_degree[:, None] + normalized_degree[None, :]) * 0.5
        degree_penalty *= scaled_usage_weight * 0.3  # small additional bias
        total_penalty += degree_penalty
    
    # Final penalized distance matrix
    penalized = edge_distance + total_penalty
    
    # Ensure finiteness and symmetry
    penalized = np.where(np.isfinite(penalized), penalized, np.inf)
    # Replace any remaining non-finite values with a large finite number
    large_val = np.max(edge_distance) * 10 + 1.0
    penalized = np.where(np.isfinite(penalized), penalized, large_val)
    
    # Enforce symmetry
    penalized = (penalized + penalized.T) / 2.0
    
    # Zero out diagonal (self-loops should not be penalized)
    np.fill_diagonal(penalized, 0.0)
    
    # Deadline guard: if we're close to the limit, return immediately
    if time.time() - _start_time > _DEADLINE:
        return penalized
    
    return penalized
# EVOLVE-BLOCK-END
