"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty for TSP.
    
    Uses an adaptive additive penalty that scales with:
    1. Edge usage count (historical)
    2. Whether the edge is in the current local optimum tour
    3. A stagnation-aware multiplier based on how many times edges have been reused
    """
    n = edge_distance.shape[0]
    
    # Base distance matrix
    penalized = edge_distance.copy()
    
    # Compute scale for penalties relative to the problem's distance scale
    # Use median of non-zero distances for robustness
    nonzero = edge_distance[edge_distance > 0]
    if len(nonzero) > 0:
        dist_scale = np.median(nonzero)
    else:
        dist_scale = 1.0
    
    # Penalty for edges in the current local optimum tour
    # These are the edges we most want to break
    tour_edges = set()
    for i in range(n):
        j = local_opt_tour[i]
        k = local_opt_tour[(i + 1) % n]
        # The tour is a sequence of nodes; edges connect consecutive nodes
        # But we need to interpret local_opt_tour correctly
        # It's a tour: sequence of node indices
        # Edge (i, j) where j = local_opt_tour[(i+1) % n]
        pass
    
    # Actually, local_opt_tour is the sequence of nodes in the tour
    # So edge between local_opt_tour[i] and local_opt_tour[(i+1)%n]
    tour_edge_set = set()
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_edge_set.add((min(a, b), max(a, b)))
    
    # Adaptive penalty: stronger when edges are frequently reused
    # edge_n_used accumulates counts over rounds
    max_used = edge_n_used.max() if edge_n_used.size > 0 else 0
    avg_used = edge_n_used.mean() if edge_n_used.size > 0 else 0
    
    # Stagnation indicator: if many edges have high usage, we're stuck
    # Use a smooth function to increase penalty with stagnation
    if max_used > 0:
        stagnation_factor = 1.0 + 0.5 * (avg_used / (1.0 + max_used))
    else:
        stagnation_factor = 1.0
    
    # Base penalty weight
    base_penalty = 0.1 * dist_scale
    
    # Apply penalty proportional to usage count, with saturation
    # Use a concave function so heavily used edges get more penalty but not unboundedly
    # penalty = base_penalty * stagnation_factor * log(1 + edge_n_used)
    usage_penalty = base_penalty * stagnation_factor * np.log1p(edge_n_used)
    
    # Add extra penalty for edges in the current tour
    # This ensures we actively try to break the current local optimum
    tour_penalty_weight = 0.15 * dist_scale * stagnation_factor
    for (a, b) in tour_edge_set:
        usage_penalty[a, b] += tour_penalty_weight
        usage_penalty[b, a] += tour_penalty_weight
    
    # Combine: penalized distance = true distance + usage penalty
    penalized += usage_penalty
    
    # Ensure finite values
    penalized = np.where(np.isfinite(penalized), penalized, edge_distance)
    
    return penalized
# EVOLVE-BLOCK-END
