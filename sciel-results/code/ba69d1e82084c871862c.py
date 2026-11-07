"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step of guided local search for TSP.
    
    Returns a modified distance matrix where edges frequently used in previous
    local optima are penalized, encouraging the subsequent 2-opt to find
    different (potentially better) tours.
    """
    start_time = time.time()
    n = edge_distance.shape[0]
    
    # Safety margin: return original if we're running out of time
    if time.time() - start_time > 280:
        return edge_distance.copy()
    
    # Initialize output with true distances
    d_out = edge_distance.copy()
    
    if n <= 1:
        return d_out
    
    # Get edges in the current local optimum tour
    tour = local_opt_tour
    if len(tour) != n:
        # Fallback: just use edge_n_used
        tour = list(range(n))
    
    # Create a set of edges in the current tour for quick lookup
    tour_edges = set()
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        if u < v:
            tour_edges.add((u, v))
        else:
            tour_edges.add((v, u))
    
    # Compute penalty based on edge usage frequency
    # Use a saturating function to avoid extreme penalties
    max_used = edge_n_used.max() if edge_n_used.size > 0 else 0
    if max_used == 0:
        # First iteration: no penalties yet, but we can still apply a small
        # perturbation to help escape potential symmetry
        return d_out
    
    # Normalize usage counts
    # Use a smooth, bounded penalty
    # penalty = alpha * (used / (used + beta))
    # where beta controls how quickly the penalty saturates
    
    # Adaptive parameters based on usage statistics
    mean_used = edge_n_used[edge_n_used > 0].mean() if np.any(edge_n_used > 0) else 1.0
    std_used = edge_n_used[edge_n_used > 0].std() if np.any(edge_n_used > 0) else 1.0
    
    # Scale alpha and beta based on the distribution of usage counts
    # Higher alpha for more aggressive escape, lower for more conservative
    alpha = 0.4  # Maximum additional penalty factor (up to 40% increase)
    beta = max(1.0, mean_used * 0.3)  # Saturation parameter tied to mean usage
    
    # Compute per-edge penalty with a smooth saturating function
    # Use a slightly different form: alpha * used / (used + beta)
    penalty = alpha * (edge_n_used / (edge_n_used + beta))
    
    # Geometry-aware repulsive force: penalize short edges more heavily
    # This forces the search to explore alternative short connections
    # Short edges are more "valuable" and thus need stronger perturbation
    # to escape local optima that rely on them
    
    # Compute inverse distance weighting (with small epsilon to avoid division by zero)
    # Use only upper triangle to avoid double-counting
    dist_matrix = edge_distance.copy()
    np.fill_diagonal(dist_matrix, np.inf)  # Avoid division by zero on diagonal
    
    # Inverse distance: shorter edges get larger weight
    inv_dist = 1.0 / (dist_matrix + 1e-10)
    
    # Normalize inverse distances to [0, 1] range
    # Use the maximum non-diagonal value for normalization
    max_inv_dist = inv_dist.max()
    if max_inv_dist > 0:
        inv_dist_norm = inv_dist / max_inv_dist
    else:
        inv_dist_norm = np.zeros_like(inv_dist)
    
    # Geometry-aware penalty: scale the usage penalty by inverse distance
    # Short edges (high inv_dist_norm) get amplified penalties
    geometry_penalty = penalty * (0.5 + inv_dist_norm)
    
    # Additional immediate penalty for edges in the current local optimum
    # This helps escape the current local optimum more aggressively
    # Scale this based on how "stuck" we seem to be (high usage counts)
    immediate_scale = 0.06 + 0.03 * min(1.0, max_used / 15.0)
    immediate_penalty = np.zeros_like(edge_n_used)
    for (u, v) in tour_edges:
        # Apply geometry-aware immediate penalty: short tour edges get more penalty
        edge_dist = edge_distance[u, v]
        # Normalize edge distance relative to median edge length
        median_edge_len = np.median(edge_distance[~np.eye(n, dtype=bool)])
        if median_edge_len > 0:
            dist_factor = 1.0 / (edge_dist / median_edge_len + 1e-10)
            dist_factor = min(dist_factor, 3.0)  # Cap the factor
        else:
            dist_factor = 1.0
        immediate_penalty[u, v] += immediate_scale * dist_factor
        immediate_penalty[v, u] += immediate_scale * dist_factor
    
    # Combine penalties
    total_penalty = geometry_penalty + immediate_penalty
    
    # Apply penalty multiplicatively
    d_out = edge_distance * (1.0 + total_penalty)
    
    # Ensure the matrix is symmetric
    d_out = (d_out + d_out.T) / 2.0
    
    # Ensure diagonal is zero
    np.fill_diagonal(d_out, 0.0)
    
    # Ensure all values are finite
    d_out[~np.isfinite(d_out)] = edge_distance.max()
    
    return d_out
# EVOLVE-BLOCK-END
