"""Guided local search penalty rule for the traveling salesman problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Strategy:
    1. Penalize edges in the current local-opt tour to force escape.
    2. Scale penalty by cumulative usage (edge_n_used) to target
       frequently-reused edges more strongly.
    3. Use a smooth, bounded penalty that preserves metric-like properties
       enough for 2-opt to work effectively.
    """
    n = edge_distance.shape[0]
    
    # Time guard: ensure we finish well within 300s
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Build mask of edges in the current local-opt tour
    tour_mask = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_mask[a, b] = 1.0
        tour_mask[b, a] = 1.0
    
    # Normalize usage with a saturating function to avoid extreme values
    max_used = np.max(edge_n_used)
    if max_used > 0:
        # Use a smooth saturating function: usage / (1 + usage)
        # This gives values in (0, 1) and saturates for large usage
        normalized_usage = edge_n_used / (1.0 + edge_n_used)
    else:
        normalized_usage = np.zeros_like(edge_n_used)
    
    # Penalty design:
    # - Base penalty for all current tour edges: ensures escape
    # - Usage-dependent boost: edges used more often get more penalty
    # - The penalty is applied multiplicatively
    
    # Adaptive base penalty: start moderate, increase slightly with usage
    if max_used > 0:
        # Base penalty grows slowly with usage to encourage stronger escape later
        base_penalty = 0.08 + 0.015 * min(max_used, 15.0)
        # Usage boost: additional penalty scaled by normalized usage
        usage_boost = 0.15
    else:
        base_penalty = 0.08
        usage_boost = 0.15
    
    # Compute penalty only for tour edges
    # The penalty factor: 1 + penalty, applied multiplicatively
    tour_penalty = (base_penalty + usage_boost * normalized_usage) * tour_mask
    
    # Apply penalty multiplicatively
    result = edge_distance * (1.0 + tour_penalty)
    
    # Ensure finite values and positive distances
    result = np.nan_to_num(result, nan=edge_distance, posinf=edge_distance, neginf=0.0)
    
    # Ensure diagonal is zero (distance from node to itself)
    np.fill_diagonal(result, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return result
    
    return result
# EVOLVE-BLOCK-END
