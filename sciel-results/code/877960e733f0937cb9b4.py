"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP using additive threshold-based penalties.
    
    Args:
        edge_distance: true Euclidean distance matrix (n x n)
        local_opt_tour: current tour as a sequence of node indices
        edge_n_used: symmetric count matrix of how often each edge has been used
    
    Returns:
        Perturbed distance matrix with additive penalties applied to frequently used edges.
    """
    n = edge_distance.shape[0]
    
    # Get usage counts for edges in the current tour
    tour_len = len(local_opt_tour)
    
    # Compute a threshold: only penalize edges that have been used more than 
    # the median usage of tour edges
    tour_edge_usage = np.zeros(tour_len)
    for i in range(tour_len):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % tour_len]
        tour_edge_usage[i] = edge_n_used[u, v]
    
    if tour_len > 0 and np.any(tour_edge_usage > 0):
        threshold = np.median(tour_edge_usage[tour_edge_usage > 0])
    else:
        threshold = 0
    
    # Additive penalty: penalty = alpha * max(0, usage - threshold) * (1 + beta * distance)
    # This makes penalties proportional to edge length, so short edges (which are more 
    # valuable) get relatively larger perturbations
    alpha = 0.12
    beta = 2.0
    
    # Compute excess usage above threshold
    excess_usage = np.maximum(0, edge_n_used - threshold)
    
    # Scale penalty by edge distance to make it relative
    # Use (1 + beta * dist) so that longer edges get slightly more penalty in absolute terms
    # but the relative effect is stronger on shorter edges
    distance_scale = 1.0 + beta * edge_distance
    
    # Additive penalty
    additive_penalty = alpha * excess_usage * distance_scale
    
    # Apply only to off-diagonal elements
    perturbed = edge_distance + additive_penalty
    
    # Ensure diagonal remains zero
    np.fill_diagonal(perturbed, 0.0)
    
    # Ensure symmetry
    perturbed = (perturbed + perturbed.T) / 2.0
    
    # Ensure all values are finite and positive
    perturbed = np.where(np.isfinite(perturbed) & (perturbed > 0), perturbed, edge_distance)
    
    return perturbed
# EVOLVE-BLOCK-END
