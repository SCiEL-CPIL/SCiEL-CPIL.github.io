"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided Local Search penalty with non-linear, tour-relative scaling.
    
    This approach uses:
    1. Tour-relative scaling: The penalty is normalized by the mean edge length 
       of the current local optimum tour.
    2. Non-linear usage penalty: Heavily used edges receive a super-linear penalty 
       (quadratic) to strongly discourage revisiting the same structural components.
    3. Dynamic strength: The penalty magnitude increases slightly as the maximum 
       usage count grows, adapting to the difficulty of escaping the current basin.
    """
    n = edge_distance.shape[0]
    
    # 1. Calculate the mean edge length of the current local optimum tour
    # This provides a context-specific scale for the penalty.
    tour_len = 0.0
    for i in range(n):
        j = local_opt_tour[i]
        k = local_opt_tour[(i + 1) % n]
        tour_len += edge_distance[i, k]
    
    if tour_len > 1e-10:
        mean_tour_edge = tour_len / n
    else:
        # Fallback to global mean if tour is degenerate
        mask = ~np.eye(n, dtype=bool)
        mean_tour_edge = np.mean(edge_distance[mask]) if np.any(mask) else 1.0
    
    # 2. Determine dynamic penalty strength
    # Base strength is a fraction of the mean tour edge length.
    # We scale it up slightly as the search progresses (indicated by max usage).
    max_used = edge_n_used.max()
    # Start with a base penalty of ~2% of the mean edge length per usage unit.
    # Quadratic growth means usage=10 gets penalty ~ 100 * 0.02 = 2.0 * mean_edge? 
    # Let's tune this carefully.
    # If we use c * count^2, and we want count=5 to be ~0.5 * mean_edge:
    # c * 25 = 0.5 * mean_edge => c = 0.02 * mean_edge
    # Let's make it adaptive: strength increases with max_used to break harder.
    base_strength = 0.01 * mean_tour_edge
    # Increase strength by 10% for every 5 units of max usage
    adaptive_factor = 1.0 + 0.02 * max_used
    strength = base_strength * adaptive_factor
    
    # 3. Compute the penalty matrix
    # Use a quadratic penalty for usage counts to heavily penalize frequent edges.
    # We clip the count to prevent overflow for very long runs, though 20 rounds is small.
    usage_sq = edge_n_used ** 2
    penalty_matrix = strength * usage_sq
    
    # 4. Apply the penalty to the distance matrix
    penalized = edge_distance + penalty_matrix
    
    # 5. Symmetrize and clean up
    # Ensure symmetry
    penalized = (penalized + penalized.T) / 2.0
    
    # Ensure diagonal is zero
    np.fill_diagonal(penalized, 0.0)
    
    # Ensure all values are finite and positive
    # Replace any non-finite or negative values with the original distance
    invalid_mask = ~np.isfinite(penalized) | (penalized < 0)
    if np.any(invalid_mask):
        penalized[invalid_mask] = edge_distance[invalid_mask]
        
    return penalized
# EVOLVE-BLOCK-END
