"""Guided local search penalty rule for the traveling salesman problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Strategy:
    1. Add an additive penalty to edges in the current local-opt tour.
    2. Scale penalty by cumulative usage count (edge_n_used) so that
       frequently-reused edges are penalized more strongly.
    3. The penalty is additive and bounded, preserving the relative
       geometry of non-tour edges for effective 2-opt.
    """
    n = edge_distance.shape[0]
    
    # Time guard
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Build mask of edges in the current local-opt tour
    tour_mask = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_mask[a, b] = 1.0
        tour_mask[b, a] = 1.0
    
    # Infer round number from max usage
    max_used = float(np.max(edge_n_used))
    
    # Additive penalty design:
    # - Base penalty: moderate constant to force escape from current tour
    # - Usage-scaled penalty: edges used more often get additional penalty
    # - Clip usage to avoid runaway penalties in later rounds
    
    # Base penalty: strong enough to make tour edges unattractive
    base_penalty = 0.15
    
    # Usage scaling: clip to [0, 20] to bound the penalty
    clipped_usage = np.clip(edge_n_used, 0.0, 20.0)
    
    # Penalty per unit usage: each time an edge is reused, add this much
    usage_rate = 0.08
    
    # Total additive penalty for tour edges
    penalty = (base_penalty + usage_rate * clipped_usage) * tour_mask
    
    # Apply additive penalty
    result = edge_distance + penalty
    
    # Ensure finite values
    result = np.nan_to_num(result, nan=edge_distance, posinf=edge_distance, neginf=0.0)
    
    # Ensure diagonal is zero
    np.fill_diagonal(result, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return result
    
    return result
# EVOLVE-BLOCK-END
