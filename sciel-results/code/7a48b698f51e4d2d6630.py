"""Guided local search penalty rule for the traveling salesman problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Strategy:
    - Apply a high multiplicative penalty ONLY to edges in the current local-optimal tour.
    - All other edges retain their true distances, preserving relative geometry.
    - Penalty strength scales with cumulative usage of each edge (edge_n_used).
    - This forces 2-opt to sever specific tour connections without distorting the rest.
    
    The penalty factor for a tour edge (i,j) is:
        factor = 1 + base + boost * normalized_usage(i,j)
    where normalized_usage = usage / (1 + usage) in (0, 1).
    """
    n = edge_distance.shape[0]
    
    # Time guard
    start_time = time.time()
    deadline = start_time + 250.0
    
    # Build mask of edges in the current local-opt tour
    tour_mask = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_mask[a, b] = 1.0
        tour_mask[b, a] = 1.0
    
    # Normalize usage with a saturating function
    max_used = float(np.max(edge_n_used)) if edge_n_used.size > 0 else 0.0
    if max_used > 0:
        normalized_usage = edge_n_used / (1.0 + edge_n_used)
    else:
        normalized_usage = np.zeros_like(edge_n_used)
    
    # Adaptive penalty parameters:
    # - Base penalty: moderate, ensures all tour edges are penalized
    # - Boost: additional penalty for frequently-used edges
    # As rounds progress (max_used grows), we increase base penalty to encourage
    # stronger escape from increasingly stale local optima.
    round_proxy = min(max_used, 20.0)
    base_penalty = 0.15 + 0.015 * round_proxy
    usage_boost = 0.30 + 0.01 * round_proxy
    
    # Compute penalty factor for tour edges only
    # factor = 1 + (base_penalty + usage_boost * normalized_usage)
    penalty_factor = 1.0 + (base_penalty + usage_boost * normalized_usage) * tour_mask
    
    # Apply multiplicatively: only tour edges get scaled up
    result = edge_distance * penalty_factor
    
    # Ensure finite values
    result = np.nan_to_num(result, nan=np.max(edge_distance) if edge_distance.size > 0 else 1.0,
                           posinf=np.max(edge_distance) if edge_distance.size > 0 else 1.0,
                           neginf=0.0)
    
    # Ensure diagonal is zero
    np.fill_diagonal(result, 0.0)
    
    # Ensure symmetry (should already be symmetric, but guard against floating point)
    result = (result + result.T) * 0.5
    
    # Check deadline
    if time.time() > deadline:
        return result
    
    return result
# EVOLVE-BLOCK-END
