import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided Local Search penalty for TSP.
    
    Applies adaptive penalties to edges based on their accumulated usage,
    scaled relative to the problem's distance distribution.
    """
    n = edge_distance.shape[0]
    
    # Compute scale factor: median of non-diagonal distances
    mask = ~np.eye(n, dtype=bool)
    nonzero_mask = (edge_distance > 1e-10) & mask
    if np.any(nonzero_mask):
        dist_scale = np.median(edge_distance[nonzero_mask])
    else:
        dist_scale = 1.0
    
    # Ensure dist_scale is positive and reasonable
    if dist_scale <= 0 or not np.isfinite(dist_scale):
        dist_scale = 1.0
    
    # Use a smooth, saturating penalty based on usage count
    # log(1 + usage) grows slowly, preventing extreme penalties
    # Scale by dist_scale so penalty magnitude is comparable to edge lengths
    # The factor 0.15 controls overall penalty strength
    
    usage = edge_n_used.astype(np.float64)
    
    # Soft penalty: log(1+usage) gives diminishing returns for high usage
    penalty_strength = 0.15 * dist_scale * np.log1p(usage)
    
    # Apply as multiplicative penalty: multiply distances by (1 + penalty)
    # This keeps all distances positive and finite
    penalty_matrix = 1.0 + penalty_strength / dist_scale  # normalized to be O(1)
    
    # Actually, let's apply the penalty directly as an additive term scaled appropriately
    # Additive penalty: edge_distance + penalty_strength
    # But multiplicative is more stable. Let's use:
    # new_dist = edge_distance * (1 + alpha * log1p(usage) / dist_scale)
    
    # Simplified: normalize usage by max to get [0,1] range, then apply
    max_usage = usage.max()
    if max_usage > 0:
        normalized_usage = usage / max_usage
    else:
        normalized_usage = usage
    
    # Penalty factor: 1 + beta * normalized_usage * (dist_scale scale factor)
    # Use beta = 0.2 for moderate penalty
    beta = 0.2
    penalty_factor = 1.0 + beta * normalized_usage
    
    # Apply penalty
    result = edge_distance * penalty_factor
    
    # Ensure finiteness and non-negativity
    result = np.where(np.isfinite(result), result, edge_distance)
    result = np.maximum(result, 1e-10)
    
    return result
