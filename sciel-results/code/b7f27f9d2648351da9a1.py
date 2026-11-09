<<<<<<< SEARCH
def _soft_min_value_and_grad(centres, beta=200.0):
    """
    Compute soft-min of all constraint values (pairwise distances/2 and boundary clearances)
    and its gradient. This provides a smooth surrogate for the hard min.
    
    Returns:
        soft_min_val: the soft-min value
        grad: gradient of the soft-min w.r.t. centres (shape (N, 3))
    """
    n = len(centres)
    
    # Collect all constraint values and their gradients
    # Boundary constraints: g_ik = 0.5 - |c_ik|
    # Pairwise constraints: g_ij = |c_i - c_j| / 2
    
    # For efficiency, compute soft-min using log-sum-exp trick on negatives
    # soft_min(x) = -logsumexp(-beta * x) / beta
    
    # Compute all constraint values
    # Boundary: for each point i and coordinate k
    d_boundary = 0.5 - np.abs(centres)  # shape (n, 3)
    
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # We'll build a list of (value, gradient_contribution)
    # But for memory efficiency with N=100, let's use a different approach
    
    # Use the fact that soft-min is dominated by the smallest few values
    # First, find approximate minimum
    min_boundary = np.min(d_boundary)
    min_pair = np.min(dists)
    r_approx = min(min_pair / 2.0, min_boundary)
    
    # Only consider constraints that are close to the minimum
    tol = 0.05  # tolerance for "active" constraints
    
    # Build gradient
    grad = np.zeros_like(centres)
    
    # Collect active constraint weights
    # For soft-min: weight_i = exp(-beta * (x_i - x_min)) / sum(exp(-beta * (x_j - x_min)))
    
    # To avoid underflow/overflow, shift by the minimum
    # We'll compute weights for boundary and pairwise separately, then combine
    
    # Boundary constraints
    # For each (i, k), value = d_boundary[i, k]
    # Gradient contribution to c_ik: -sign(c_ik) if c_ik != 0, else 0
    # Actually d/dc_ik (0.5 - |c_ik|) = -sign(c_ik) for c_ik != 0
    
    b_vals = d_boundary.ravel()  # shape (3n,)
    b_idx = np.argmin(b_vals)
    b_min = b_vals[b_idx]
    
    # Pairwise constraints
    p_vals = (dists / 2.0)
    np.fill_diagonal(p_vals, np.inf)
    p_idx_flat = np.argmin(p_vals)
    p_min = p_vals.ravel()[p_idx_flat]
    
    # Combine: we have two sets of constraints
    # Let's compute the soft-min over all constraints together
    
    # For numerical stability, find global min
    all_min = min(b_min, p_min)
    
    # Compute weights for boundary constraints
    b_shifted = b_vals - all_min
    b_weights = np.exp(-beta * b_shifted)
    
    # Compute weights for pairwise constraints
    p_vals_flat = p_vals.ravel()
    p_shifted = p_vals_flat - all_min
    p_weights = np.exp(-beta * p_shifted)
    
    # Total weight
    total_weight = b_weights.sum() + p_weights.sum()
    
    if total_weight < 1e-300:
        # Degenerate case, fall back to hard min
        return r_approx, grad
    
    # Soft-min value
    soft_min_val = all_min + np.log(total_weight) / beta
    
    # Compute gradient from boundary constraints
    # For boundary constraint (i, k): value = 0.5 - |c_ik|
    # d(value)/d(c_ik) = -sign(c_ik) if c_ik != 0, 0.5 if c_ik == 0 (subgradient)
    # Weight for this constraint: b_weights[i*3+k] / total_weight
    # Contribution to grad[i, k]: weight * (-sign(c_ik))
    
    b_weights_reshaped = b_weights.reshape(n, 3)
    signs = np.sign(centres)
    # Handle c_ik = 0: use 0 as subgradient (or random, but 0 is safer)
    grad_boundary = -b_weights_reshaped * signs
    grad_boundary /= total_weight
    
    # Compute gradient from pairwise constraints
    # For pairwise constraint (i, j): value = |c_i - c_j| / 2
    # d(value)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|) if |c_i - c_j| > 0
    # d(value)/d(c_j) = (c_j - c_i) / (2 * |c_i - c_j|) = -(c_i - c_j) / (2 * |c_i - c_j|)
    
    # p_weights is over the flattened upper triangle (or full matrix with diagonal inf)
    # Let's iterate over pairs
    grad_pair = np.zeros_like(centres)
    for i in range(n):
        for j in range(i + 1, n):
            w = p_weights[i * n + j] / total_weight
            if w < 1e-15:
                continue
            d = dists[i, j]
            if d < 1e-12:
                continue
            unit = (centres[i] - centres[j]) / d
            grad_pair[i] += w * unit / 2.0
            grad_pair[j] -= w * unit / 2.0
    
    grad = grad_boundary + grad_pair
    
    return soft_min_val, grad
=======
def _soft_min_value_and_grad(centres, beta=200.0):
    """
    Compute soft-min of all constraint values (pairwise distances/2 and boundary clearances)
    and its gradient. This provides a smooth surrogate for the hard min.
    Vectorized pairwise gradient computation for speed.
    
    Returns:
        soft_min_val: the soft-min value
        grad: gradient of the soft-min w.r.t. centres (shape (N, 3))
    """
    n = len(centres)
    
    # Boundary constraints: g_ik = 0.5 - |c_ik|
    d_boundary = 0.5 - np.abs(centres)  # shape (n, 3)
    
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Approximate minimum
    min_boundary = np.min(d_boundary)
    min_pair = np.min(dists)
    r_approx = min(min_pair / 2.0, min_boundary)
    
    # Find global min for numerical stability
    b_vals = d_boundary.ravel()  # shape (3n,)
    b_min = np.min(b_vals)
    
    p_vals = dists / 2.0
    np.fill_diagonal(p_vals, np.inf)
    p_min = np.min(p_vals)
    
    all_min = min(b_min, p_min)
    
    # Compute weights for boundary constraints
    b_shifted = b_vals - all_min
    b_weights = np.exp(-beta * b_shifted)
    
    # Compute weights for pairwise constraints
    p_shifted = p_vals - all_min
    p_weights = np.exp(-beta * p_shifted)
    
    # Total weight
    total_weight = b_weights.sum() + p_weights.sum()
    
    if total_weight < 1e-300:
        return r_approx, np.zeros_like(centres)
    
    # Soft-min value
    soft_min_val = all_min + np.log(total_weight) / beta
    
    # Gradient from boundary constraints
    b_weights_reshaped = b_weights.reshape(n, 3)
    signs = np.sign(centres)
    # Where centre is exactly 0, use 0 as subgradient
    signs[centres == 0] = 0
    grad_boundary = -b_weights_reshaped * signs / total_weight
    
    # Vectorized gradient from pairwise constraints
    # For each pair (i,j): weight w_ij, direction unit_ij = (c_i - c_j) / |c_i - c_j|
    # grad[i] += w_ij * unit_ij / 2
    # grad[j] -= w_ij * unit_ij / 2
    # 
    # Using matrix operations:
    # Let W be the weight matrix (upper triangle only, symmetric extension for computation)
    # Let U be the unit direction matrix: U[i,j] = (c_i - c_j) / dists[i,j]
    # Then grad_pair[i] = sum_j W[i,j] * U[i,j] / 2
    # But we only want upper triangle, so let's be careful.
    
    # Compute unit directions
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_dirs = diff / safe_dists[:, :, np.newaxis]
    unit_dirs[dists < 1e-12] = 0.0
    
    # Weight matrix (full, but diagonal is 0 since p_weights diagonal is exp(-beta*inf)=0)
    w_matrix = p_weights / total_weight
    
    # For each i, grad_pair[i] = sum_j w_matrix[i,j] * unit_dirs[i,j] / 2
    # This is a matrix-vector product-like operation
    # grad_pair[i, k] = sum_j w_matrix[i,j] * unit_dirs[i,j,k] / 2
    
    # Use einsum for efficiency
    # grad_pair = (w_matrix @ unit_dirs) / 2
    # But w_matrix is not symmetric in the way we want - we want each pair counted once
    # Actually, since unit_dirs[i,j] = -unit_dirs[j,i] and w_matrix[i,j] = w_matrix[j,i],
    # the sum over all j naturally gives the correct result if we use the full matrix.
    # Let's verify: for pair (i,j), contribution to grad[i] is w_ij * unit_ij / 2
    #              contribution to grad[j] is w_ji * unit_ji / 2 = w_ij * (-unit_ij) / 2
    # So using full symmetric weight matrix and full unit matrix works correctly.
    
    grad_pair = np.einsum('ij,ijk->ik', w_matrix, unit_dirs) / 2.0
    
    grad = grad_boundary + grad_pair
    
    return soft_min_val, grad
>>>>>>> REPLACE

