"""Guided local search penalty rule for the traveling salesman problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Strategy:
    1. Repulsive force model: penalty inversely proportional to edge length.
       Short edges get disproportionately large penalties, creating a strong
       gradient to escape local optima by exploring alternative short connections.
    2. Usage-based penalty with non-linear scaling (power 1.5) for more aggressive
       escape from frequently-used edge patterns.
    3. Tour-edge penalty: edges in the current local-opt tour are penalized
       to force 2-opt to break away from the current solution.
    4. All penalties are applied multiplicatively to keep distances positive.
    """
    start_time = time.time()
    n = edge_distance.shape[0]
    
    # Deadline guard
    deadline = start_time + 280.0
    
    # Compute median edge length for normalization
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    upper_edges = edge_distance[mask]
    median_dist = np.median(upper_edges)
    if median_dist <= 0:
        median_dist = 1.0
    
    # 1. Repulsive force model: smooth inverse-distance penalty
    # Use a softer formulation that still penalizes short edges disproportionately
    # but with less extreme gradient
    normalized_dist = edge_distance / median_dist
    # Smooth repulsive force: use (1 + d)^(-0.8) for a softer tail
    # This still gives higher penalty to short edges but less extreme than 1/(1+d)
    repulsive_alpha = 0.35
    repulsive_penalty = repulsive_alpha * np.power(1.0 + normalized_dist, -0.8)
    
    # 2. Usage-based penalty with adaptive scaling
    # Scale based on the 95th percentile of usage to adapt to the actual distribution
    alpha_usage = 0.22
    clipped_usage = np.clip(edge_n_used, 0, 100)
    # Use 95th percentile as reference for adaptive scaling
    usage_ref = np.percentile(clipped_usage[clipped_usage > 0], 95) if np.any(clipped_usage > 0) else 1.0
    usage_ref = max(usage_ref, 1.0)
    # Normalize usage relative to the reference, then apply power scaling
    normalized_usage = clipped_usage / usage_ref
    usage_penalty = alpha_usage * np.power(normalized_usage / (1.0 + normalized_usage), 1.3)
    
    # 3. Tour-edge penalty: force breaking from current tour
    # Increase slightly to ensure stronger escape from current local optimum
    immediate_scale = 0.22
    tour_mask = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_mask[a, b] = 1.0
        tour_mask[b, a] = 1.0
    
    immediate_penalty = immediate_scale * tour_mask
    
    # Check deadline
    if time.time() > deadline:
        result = edge_distance * (1.0 + immediate_penalty)
        result = np.nan_to_num(result, nan=edge_distance, posinf=edge_distance, neginf=0.0)
        return result
    
    # Combine all penalties
    total_penalty = repulsive_penalty + usage_penalty + immediate_penalty
    
    # Apply multiplicatively to keep all distances positive and finite
    result = edge_distance * (1.0 + total_penalty)
    
    # Ensure finite values
    result = np.nan_to_num(result, nan=edge_distance, posinf=edge_distance, neginf=0.0)
    
    # Final deadline check
    if time.time() > deadline:
        return result
    
    return result
# EVOLVE-BLOCK-END
