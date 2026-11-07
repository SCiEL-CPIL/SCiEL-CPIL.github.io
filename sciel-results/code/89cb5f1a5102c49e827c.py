"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation: add edge-specific penalties to edges
    in the current local optimal tour, scaled by a power of their usage count.
    
    This uses an additive, targeted penalty approach (different from multiplicative
    scaling) that specifically penalizes edges in the current tour, with the
    penalty magnitude growing non-linearly with usage count to encourage escape
    from persistent local optima.
    """
    n = edge_distance.shape[0]
    # Work on a copy to avoid modifying the input
    perturbed = edge_distance.copy()
    
    # Compute mean edge distance for scaling the penalty
    # Use upper triangle to get unique edge distances
    iu = np.triu_indices(n, k=1)
    mean_edge_dist = np.mean(edge_distance[iu])
    
    # Get the edges in the current local optimal tour
    tour = local_opt_tour
    tour_edges = []
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        if u < v:
            tour_edges.append((u, v))
        else:
            tour_edges.append((v, u))
    
    # Apply additive penalty to each edge in the tour
    # Penalty scales with a power of usage count, normalized by mean edge distance
    for (u, v) in tour_edges:
        count = edge_n_used[u, v]
        if count > 0:
            # Non-linear penalty: quadratic in count, scaled by mean edge distance
            # The factor 0.05 controls the strength of perturbation
            penalty = 0.05 * mean_edge_dist * (count ** 1.5)
            perturbed[u, v] += penalty
            perturbed[v, u] += penalty
    
    # Ensure all values are finite and non-negative
    perturbed = np.maximum(perturbed, 0.0)
    perturbed = np.where(np.isfinite(perturbed), perturbed, mean_edge_dist)
    
    return perturbed
# EVOLVE-BLOCK-END
