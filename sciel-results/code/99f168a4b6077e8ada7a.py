<<<<<<< SEARCH
def _soft_min_objective_fast(x, n_points, beta=150.0, k_neighbors=6):
    """
    Fast soft-min objective using KDTree for nearest neighbors only.
    x is flattened (N,2) array. Returns negative of soft-min radius (to minimize).
    """
    pts = x.reshape(n_points, 2)
    
    # Boundary margins
    norms = np.linalg.norm(pts, axis=1)
    boundary_margins = 1.0 - norms
    
    # Use KDTree to get nearest k neighbors for each point
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=k_neighbors + 1)  # +1 for self
    # Exclude self-distance (first column)
    pair_dists = dists[:, 1:] / 2.0  # distance / 2 = radius contribution
    
    # Soft min of pair distances (only over nearest neighbors)
    min_pair = np.min(pair_dists)
    # Shift for numerical stability
    exp_terms = np.exp(-beta * (pair_dists - min_pair))
    soft_min_pair = min_pair - np.log(np.sum(exp_terms)) / beta
    
    # Soft min of boundary margins
    min_boundary = np.min(boundary_margins)
    exp_terms_b = np.exp(-beta * (boundary_margins - min_boundary))
    soft_min_boundary = min_boundary - np.log(np.sum(exp_terms_b)) / beta
    
    # Soft min of the two components
    m = min(soft_min_pair, soft_min_boundary)
    exp_p = np.exp(-beta * (soft_min_pair - m))
    exp_b = np.exp(-beta * (soft_min_boundary - m))
    soft_min_radius = m - np.log(exp_p + exp_b) / beta
    
    return -soft_min_radius
=======
def _soft_min_objective_smooth(x, n_points, beta=100.0):
    """
    Smooth soft-min objective using Log-Sum-Exp for differentiability.
    x is flattened (N,2) array. Returns negative of soft-min radius (to minimize).
    """
    pts = x.reshape(n_points, 2)
    
    # Boundary margins
    norms = np.linalg.norm(pts, axis=1)
    boundary_margins = 1.0 - norms
    
    # Pair distances: We need to approximate the global minimum pair distance smoothly.
    # Using KDTree is not differentiable. 
    # However, for L-BFGS-B, we can use a finite-difference gradient or just rely on the smoothness of the soft-min over a subset?
    # Actually, scipy.optimize.minimize with L-BFGS-B can use finite differences if jac is not provided.
    # But finite differences on 600 variables is expensive.
    # Let's try to construct a differentiable approximation of the min pair distance.
    # A common trick is to use the soft-min over all pairs, but that is O(N^2).
    # With N=300, N^2 = 90,000. This is computationally feasible for the objective function itself, 
    # but the gradient calculation via finite differences will be 600 * 90,000 = 54,000,000 ops per gradient step.
    # This might be too slow for 100+ iterations.
    
    # Alternative: Use the "nearest neighbor" approximation but make it smooth?
    # KDTree is not smooth.
    
    # Let's stick to the previous strategy but ensure the objective is smooth enough for L-BFGS-B's internal gradient estimation.
    # The previous _soft_min_objective_fast used np.min which is non-differentiable.
    # L-BFGS-B with finite differences will struggle with kinks.
    
    # Let's use a global soft-min over all pairs.
    # Distance matrix D
    # D[i,j] = ||p_i - p_j||
    # We want soft_min(D / 2).
    
    # Compute all pairwise distances
    # This is O(N^2) memory and time.
    # N=300 -> 300x300 matrix.
    
    # Vectorized distance calculation
    # p_i - p_j
    # (N, 1, 2) - (1, N, 2) -> (N, N, 2)
    # This creates a 300x300x2 array, which is fine.
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal (self-distance) with infinity
    dists[np.diag_indices_from(dists)] = np.inf
    
    pair_radii = dists / 2.0
    
    # Soft min of pair_radii
    min_pair = np.min(pair_radii)
    # Subtract min for stability
    shifted_pair = pair_radii - min_pair
    # Clip to avoid overflow in exp
    shifted_pair = np.clip(shifted_pair, 0, 100) 
    exp_pair = np.exp(-beta * shifted_pair)
    # Sum over all pairs
    sum_exp_pair = np.sum(exp_pair)
    # Soft min formula: min - (1/beta) * log(sum(exp(-beta*(x-min))))
    soft_min_pair = min_pair - np.log(sum_exp_pair) / beta
    
    # Soft min of boundary margins
    min_boundary = np.min(boundary_margins)
    shifted_boundary = boundary_margins - min_boundary
    shifted_boundary = np.clip(shifted_boundary, 0, 100)
    exp_boundary = np.exp(-beta * shifted_boundary)
    sum_exp_boundary = np.sum(exp_boundary)
    soft_min_boundary = min_boundary - np.log(sum_exp_boundary) / beta
    
    # Soft min of the two components
    m = min(soft_min_pair, soft_min_boundary)
    shifted_p = soft_min_pair - m
    shifted_b = soft_min_boundary - m
    exp_p = np.exp(-beta * shifted_p)
    exp_b = np.exp(-beta * shifted_b)
    soft_min_radius = m - np.log(exp_p + exp_b) / beta
    
    return -soft_min_radius
>>>>>>> REPLACE

