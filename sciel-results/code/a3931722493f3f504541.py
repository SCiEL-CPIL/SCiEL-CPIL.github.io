"""Guided local search penalty rule for the traveling salesman problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Strategy:
    1. Apply an additive penalty to edges based on their cumulative usage 
       (edge_n_used). This is the core of Guided Local Search: penalize 
       edges that have been part of previous local optima to drive the 
       search away from them.
    2. Apply an additional stronger penalty to edges in the *current* 
       local-opt tour to ensure immediate escape from the current state.
    3. Use a smooth, saturating function for the usage penalty to avoid 
       extreme values while still discriminating between frequently used 
       and rarely used edges.
    """
    n = edge_distance.shape[0]
    
    # Time guard: ensure we finish well within 300s
    start_time = time.time()
    deadline = start_time + 280.0

    # 1. Usage-based penalty for ALL edges
    # Standard GLS penalizes edges that have been in local optima.
    # edge_n_used accumulates these counts.
    # We use a smooth function: penalty = alpha * usage / (1 + usage)
    # This saturates at alpha, preventing infinite penalties.
    # However, for edges used many times, we might want a stronger effect.
    # Let's try: penalty = alpha * sqrt(usage) or alpha * log(1+usage).
    # The square root grows slower than linear but faster than log.
    # Let's stick to a linear-like growth but capped or smoothed.
    # Actually, a simple linear penalty with a cap is often robust.
    # But the prompt suggests "smooth surrogate". 
    # Let's use: P_usage = alpha * (1 - exp(-beta * usage))
    # This is smooth, bounded by alpha, and grows with usage.
    
    max_used = float(np.max(edge_n_used)) if edge_n_used.size > 0 else 0.0
    
    # Adaptive parameters
    # alpha: max penalty for any edge
    # beta: rate of increase
    # We want the penalty to be significant compared to edge lengths (approx 0.1-0.5 in unit square).
    # Let's set alpha to be around 0.2-0.5.
    
    if max_used > 0:
        # Increase alpha slightly as we go deeper into the search to force more drastic changes?
        # Or keep it constant? Standard GLS often keeps it constant or increases it if stuck.
        # Here we don't know if we are stuck, but we know we are iterating.
        alpha = 0.25
        beta = 0.5
    else:
        alpha = 0.25
        beta = 0.5

    # Compute usage penalty for all edges
    # P = alpha * (1 - exp(-beta * usage))
    # This is 0 for usage=0, and approaches alpha for large usage.
    usage_penalty = alpha * (1.0 - np.exp(-beta * edge_n_used))
    
    # 2. Additional penalty for edges in the CURRENT tour
    # This ensures we don't just stay in the current local optimum if the 
    # usage penalty isn't high enough yet (e.g., first few iterations).
    tour_mask = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_mask[a, b] = 1.0
        tour_mask[b, a] = 1.0
    
    # Base penalty for current tour edges
    # This should be strong enough to break the current tour structure.
    base_tour_penalty = 0.15
    
    total_penalty = usage_penalty + base_tour_penalty * tour_mask
    
    # Apply additive penalty
    result = edge_distance + total_penalty
    
    # Ensure symmetry (should already be symmetric if inputs are, but enforce it)
    # result = (result + result.T) / 2.0
    
    # Ensure finite values and positive distances
    result = np.nan_to_num(result, nan=edge_distance, posinf=edge_distance, neginf=0.0)
    
    # Ensure diagonal is zero
    np.fill_diagonal(result, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return result
    
    return result
# EVOLVE-BLOCK-END
