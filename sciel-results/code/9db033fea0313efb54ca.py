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
    
    # 1. Repulsive force model: penalty inversely proportional to distance
    # Short edges get large penalties, long edges get small penalties
    # Use a smooth approximation to avoid division by zero
    eps = 1e-6
    # Normalize by median to make the penalty scale-invariant
    normalized_dist = edge_distance / median_dist
    # Inverse distance penalty: 1/(1 + d) creates a smooth repulsive force
    # This gives disproportionately large penalties to short edges
    repulsive_alpha = 0.4
    repulsive_penalty = repulsive_alpha * (1.0 / (1.0 + normalized_dist))
    
    # 2. Usage-based penalty with non-linear scaling (power 1.5)
    # This creates more aggressive escape from frequently-used patterns
    alpha_usage = 0.20
    # Clip edge_n_used to avoid extreme values
    clipped_usage = np.clip(edge_n_used, 0, 50)
    usage_penalty = alpha_usage * np.power(clipped_usage / (1.0 + clipped_usage), 1.5)
    
    # 3. Tour-edge penalty: force breaking from current tour
    immediate_scale = 0.18
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
