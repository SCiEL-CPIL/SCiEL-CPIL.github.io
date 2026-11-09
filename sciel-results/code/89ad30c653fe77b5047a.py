import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided Local Search penalty for TSP.
    
    Args:
        edge_distance: (n, n) true Euclidean distance matrix
        local_opt_tour: list/array of node indices forming the current local optimum tour
        edge_n_used: (n, n) symmetric count matrix of edge usage in local optima
    
    Returns:
        (n, n) penalized distance matrix (finite, same shape)
    """
    n = edge_distance.shape[0]
    
    # Compute a characteristic scale from the distance matrix
    # Use median of non-zero distances for robustness
    upper_tri = edge_distance[np.triu_indices(n, k=1)]
    scale = np.median(upper_tri)
    if scale < 1e-10:
        scale = 1.0
    
    # Base penalty coefficient: small enough to not dominate, large enough to matter
    # alpha * count should be comparable to a fraction of typical edge length
    alpha = 0.02 * scale
    
    # Start with a copy of the true distances
    penalized = edge_distance.astype(np.float64).copy()
    
    # Additive penalty based on historical usage
    # Use a smooth, bounded function: penalty grows but saturates
    # f(count) = count / (1 + count / K) where K is a saturation parameter
    # This keeps penalties from becoming unreasonably large
    K = 5.0  # saturation: after ~5 uses, penalty growth slows
    smooth_count = edge_n_used / (1.0 + edge_n_used / K)
    
    penalized += alpha * smooth_count
    
    # Additional penalty for edges in the current local optimum tour
    # These are the edges we most want to escape
    if local_opt_tour is not None and len(local_opt_tour) > 1:
        tour = np.asarray(local_opt_tour)
        # Close the tour if not already closed
        if tour[0] != tour[-1]:
            tour = np.append(tour, tour[0])
        
        # Apply a stronger penalty to tour edges
        # The penalty should be proportional to how many times these edges have been used
        # but with an additional boost for the current tour
        for i in range(len(tour) - 1):
            u, v = tour[i], tour[i+1]
            if u != v:
                # Additional penalty: proportional to usage, with a base boost
                # Use the same smooth function but with higher alpha for current tour
                boost = 0.05 * scale * smooth_count[u, v]
                # Also add a small fixed penalty to encourage leaving the current tour
                base_boost = 0.01 * scale
                penalized[u, v] += base_boost + boost
                penalized[v, u] += base_boost + boost
    
    # Ensure symmetry
    penalized = (penalized + penalized.T) / 2.0
    
    # Ensure finite values
    penalized = np.nan_to_num(penalized, nan=0.0, posinf=1e10, neginf=0.0)
    
    # Ensure non-negative
    penalized = np.maximum(penalized, 0.0)
    
    return penalized
