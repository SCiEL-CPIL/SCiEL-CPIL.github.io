"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Strategy:
    1. Saturating penalty on edges based on cumulative usage (edge_n_used).
       This ensures that edges that have been used many times across rounds
       get increasingly discouraged, but the penalty saturates to avoid
       numerical issues.
    2. Immediate penalty on edges in the current local-opt tour to guarantee
       that the perturbed distance matrix differs from the true matrix on
       the current tour, forcing 2-opt to consider different edges.
    3. The penalty is applied multiplicatively to keep distances positive.
    """
    n = edge_distance.shape[0]
    
    # Saturating penalty based on cumulative usage
    # alpha controls the maximum penalty factor
    alpha = 0.4
    usage_penalty = alpha * edge_n_used / (1.0 + edge_n_used)
    
    # Immediate penalty on edges in the current tour
    # This ensures we always perturb the current solution
    immediate_scale = 0.12
    tour_mask = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_mask[a, b] = 1.0
        tour_mask[b, a] = 1.0
    
    immediate_penalty = immediate_scale * tour_mask
    
    # Combine penalties
    total_penalty = usage_penalty + immediate_penalty
    
    # Apply multiplicatively to keep all distances positive and finite
    result = edge_distance * (1.0 + total_penalty)
    
    # Ensure finite values
    result = np.nan_to_num(result, nan=edge_distance, posinf=edge_distance, neginf=0.0)
    
    return result
# EVOLVE-BLOCK-END
