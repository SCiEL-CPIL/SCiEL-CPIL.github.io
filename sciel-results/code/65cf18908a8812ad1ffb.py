"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided Local Search penalty: add an adaptive penalty to edges
    proportional to their accumulated usage count.
    
    The penalty strength is scaled relative to the median edge distance
    to ensure it's meaningful across different problem instances.
    """
    # Get the current tour edges for potential special handling
    n = edge_distance.shape[0]
    
    # Compute a scale factor based on the distribution of true distances
    # Use the median of non-zero, non-diagonal distances
    mask = ~np.eye(n, dtype=bool)
    nonzero_mask = (edge_distance > 1e-10) & mask
    if np.any(nonzero_mask):
        dist_scale = np.median(edge_distance[nonzero_mask])
    else:
        dist_scale = 1.0
    
    # Adaptive penalty strength: strong enough to perturb, but not degenerate
    # We want edges used ~10 times to get a penalty of ~0.5 * dist_scale
    # So strength ~ 0.05 * dist_scale
    strength = 0.05 * dist_scale
    
    # Additive penalty proportional to usage count
    penalized = edge_distance + strength * edge_n_used
    
    # Additionally, apply a multiplicative boost to edges that are in the
    # current local optimum tour and have high usage, to strongly discourage
    # returning to the same basin
    tour_edges = np.zeros_like(edge_distance)
    for i in range(n):
        j = local_opt_tour[i]
        k = local_opt_tour[(i + 1) % n]
        tour_edges[i, j] = 1.0
        tour_edges[j, i] = 1.0
        tour_edges[i, k] = 1.0
        tour_edges[k, i] = 1.0
    
    # For tour edges with high usage, add extra penalty
    max_used = edge_n_used.max()
    if max_used > 0:
        usage_frac = edge_n_used / max_used
        # Extra penalty for tour edges that are heavily used
        extra = 0.1 * dist_scale * tour_edges * usage_frac
        penalized = penalized + extra
    
    # Ensure the matrix is symmetric
    penalized = (penalized + penalized.T) / 2.0
    
    # Ensure diagonal is zero
    np.fill_diagonal(penalized, 0.0)
    
    # Ensure all values are finite
    penalized = np.where(np.isfinite(penalized), penalized, edge_distance)
    
    return penalized
# EVOLVE-BLOCK-END
