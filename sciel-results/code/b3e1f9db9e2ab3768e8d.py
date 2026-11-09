"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search using a normalized additive penalty.
    
    This approach uses an additive penalty that is normalized by the maximum usage count
    in the current matrix. This makes the perturbation strength relative to the most
    frequently used edge, providing a more stable diversification pressure compared to
    raw count-based penalties which depend on the absolute number of iterations.
    
    Args:
        edge_distance: (N, N) array of true Euclidean distances.
        local_opt_tour: (N,) array of node indices forming the current tour.
        edge_n_used: (N, N) array of symmetric counts of edge usage.
        
    Returns:
        (N, N) array of perturbed distances.
    """
    n = edge_distance.shape[0]
    if n < 2:
        return edge_distance.copy()

    # Calculate scale factor based on median distance
    lower_tri = np.tril_indices(n, k=-1)
    distances = edge_distance[lower_tri]
    non_zero_dist = distances[distances > 0]
    if len(non_zero_dist) > 0:
        median_dist = np.median(non_zero_dist)
    else:
        median_dist = 1.0

    # Get max usage count for normalization
    max_n = np.max(edge_n_used)
    
    if max_n == 0:
        return edge_distance.copy()

    # Normalize usage counts to [0, 1]
    normalized_usage = edge_n_used / max_n

    # Additive penalty coefficient.
    # We want the penalty to be significant enough to change the tour but not so large
    # that it completely ignores geometry.
    # A fraction of the median distance is a robust heuristic.
    beta = 0.5 * median_dist

    # Apply additive penalty
    perturbed_distance = edge_distance + beta * normalized_usage

    # Ensure symmetry (though input should be symmetric, floating point might drift)
    # and finiteness
    perturbed_distance = np.nan_to_num(perturbed_distance, nan=0.0, posinf=1e6, neginf=0.0)
    
    return perturbed_distance
# EVOLVE-BLOCK-END
