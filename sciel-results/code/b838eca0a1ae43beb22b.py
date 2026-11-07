"""Guided local search penalty rule for the traveling salesman problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Strategy:
    1. Smooth repulsive geometric penalty: 1/(1+d/d_med) gives bounded,
       smooth penalty that's strongest for very short edges and fades
       gracefully for longer edges.
    2. Saturating linear usage penalty: u/(1+u) provides gentle pressure
       against repeatedly-used edges without extreme values.
    3. Tour-edge penalty: moderate penalty on current tour edges to
       encourage 2-opt to explore alternative connections.
    4. All penalties applied multiplicatively to keep distances positive.
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
    
    # 1. Smooth repulsive geometric penalty
    # 1/(1 + d/d_med) is in (0, 1], strongest for d=0, weak for large d
    # Scale so that at d=median, penalty = 0.5 * alpha_geo
    normalized_dist = edge_distance / median_dist
    alpha_geo = 0.35
    geometric_penalty = alpha_geo * (1.0 / (1.0 + normalized_dist))
    
    # 2. Saturating linear usage penalty
    # u/(1+u) saturates at 1.0, providing gentle pressure
    alpha_usage = 0.22
    usage_penalty = alpha_usage * edge_n_used / (1.0 + edge_n_used)
    
    # 3. Tour-edge penalty: moderate to encourage exploration
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
    total_penalty = geometric_penalty + usage_penalty + immediate_penalty
    
    # Apply multiplicatively to keep all distances positive and finite
    result = edge_distance * (1.0 + total_penalty)
    
    # Ensure finite values
    result = np.nan_to_num(result, nan=edge_distance, posinf=edge_distance, neginf=0.0)
    
    # Final deadline check
    if time.time() > deadline:
        return result
    
    return result
# EVOLVE-BLOCK-END
