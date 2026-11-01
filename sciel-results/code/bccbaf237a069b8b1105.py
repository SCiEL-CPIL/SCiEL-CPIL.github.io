<<<<<<< SEARCH
def _grad_soft_min(x, tau=0.008):
    """
    Compute gradient of -soft_min objective numerically (for safety)
    or analytically.
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Upper triangle pairwise distances
    iu, ju = np.triu_indices(N, k=1)
    pair_upper = pair_dists[iu, ju]
    
    all_dists = np.concatenate([walls, pair_upper])
    
    # Soft-min weights: w_k = exp(-d_k/tau) / sum(exp(-d_j/tau))
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_vals = np.exp(-shifted / tau)
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(x)
    
    # Wall gradient contributions
    for i in range(N):
        w_wall = weights[i]
        # d(wall_i)/d(centers[i,0]) = -sign(centers[i,0]) if x is limiting
        if walls[i] == wall_dists_x[i]:
            grad[2*i] += w_wall * np.sign(centers[i, 0])
        if walls[i] == wall_dists_y[i]:
            grad[2*i+1] += w_wall * np.sign(centers[i, 1])
    
    # Pairwise gradient contributions
    for k in range(len(iu)):
        w_pair = weights[N + k]
        i, j = iu[k], ju[k]
        d = dists[i, j]
        if d > 1e-12:
            factor = 0.5 * w_pair / d
            diff_ij = centers[i] - centers[j]
            grad[2*i:2*i+2] -= factor * diff_ij
            grad[2*j:2*j+2] += factor * diff_ij
    
    return grad
=======
def _grad_soft_min(x, tau=0.008):
    """
    Compute gradient of -soft_min objective analytically.
    The soft-min gradient is a weighted sum of individual constraint gradients,
    where weights are proportional to exp(-d_k/tau).
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Upper triangle pairwise distances
    iu, ju = np.triu_indices(N, k=1)
    pair_upper = pair_dists[iu, ju]
    
    all_dists = np.concatenate([walls, pair_upper])
    
    # Soft-min weights: w_k = exp(-d_k/tau) / sum(exp(-d_j/tau))
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_vals = np.exp(-shifted / tau)
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(x)
    
    # Wall gradient contributions
    # d(wall_x_i)/d(c_i,0) = -sign(c_i,0), d(wall_y_i)/d(c_i,1) = -sign(c_i,1)
    # Since soft_min = min of constraints, and we minimize -soft_min,
    # the gradient of -soft_min w.r.t. position pushes away from walls.
    for i in range(N):
        w_wall = weights[i]
        # If x-wall is limiting, moving center toward wall decreases wall_dist
        # d(wall_x)/d(c_x) = -sign(c_x), so d(-soft_min)/d(c_x) includes +w*sign(c_x)
        if wall_dists_x[i] <= wall_dists_y[i]:
            grad[2*i] += w_wall * np.sign(centers[i, 0])
        else:
            grad[2*i+1] += w_wall * np.sign(centers[i, 1])
        # Also consider the other wall if it's close
        if wall_dists_y[i] <= wall_dists_x[i]:
            grad[2*i+1] += w_wall * np.sign(centers[i, 1])
        else:
            grad[2*i] += w_wall * np.sign(centers[i, 0])
    
    # Pairwise gradient contributions
    for k in range(len(iu)):
        w_pair = weights[N + k]
        i, j = iu[k], ju[k]
        d = dists[i, j]
        if d > 1e-12:
            # d(pair_dist_ij)/d(c_i) = (c_i - c_j) / |c_i - c_j|
            # d(-soft_min)/d(c_i) includes -w * (c_i - c_j)/d
            factor = 0.5 * w_pair / d
            diff_ij = centers[i] - centers[j]
            grad[2*i:2*i+2] -= factor * diff_ij
            grad[2*j:2*j+2] += factor * diff_ij
    
    return grad
>>>>>>> REPLACE

