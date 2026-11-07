"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Multiplicative distance scaling: applies a high penalty factor ONLY to
    edges in the current local-optimal tour, leaving all other edges at their
    true distances. This forces 2-opt to sever specific tour connections
    without distorting the relative geometry of the remaining search space.
    
    The penalty magnitude is modulated by cumulative usage (edge_n_used) so
    that edges repeatedly used across rounds are penalized more aggressively.
    """
    n = edge_distance.shape[0]
    
    # Start with true distances
    result = edge_distance.copy()
    
    # Build tour edge set for O(1) lookup
    tour_edges = set()
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_edges.add((a, b))
        tour_edges.add((b, a))
    
    # For each tour edge, compute a multiplicative penalty factor.
    # Base penalty is high enough to force 2-opt to break these edges.
    # Usage modulates: more-used edges get higher penalty (capped).
    base_penalty = 3.0       # minimum multiplier for tour edges
    usage_scale = 1.5        # additional multiplier per unit of usage (saturating)
    
    for (a, b) in tour_edges:
        usage = edge_n_used[a, b]
        # Saturating usage-based boost: bounded to avoid extreme values
        usage_boost = usage_scale * usage / (1.0 + usage)
        factor = base_penalty + usage_boost
        result[a, b] = edge_distance[a, b] * factor
        result[b, a] = edge_distance[b, a] * factor
    
    # Ensure finiteness
    result = np.where(np.isfinite(result), result, edge_distance)
    
    return result
# EVOLVE-BLOCK-END
