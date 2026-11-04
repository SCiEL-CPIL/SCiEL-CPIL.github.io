<<<<<<< SEARCH
def _energy_and_gradient(x, R):
    """
    Calculates the repulsion energy and gradient for a target radius R.
    Uses a smooth soft-penalty: quadratic when violated, zero otherwise.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise terms - smooth penalty for d_ij < 2R
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    dists = np.sqrt(sq_dists + 1e-15)
    np.fill_diagonal(dists, np.inf)
    np.fill_diagonal(sq_dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    d_ij = dists[i_idx, j_idx]
    
    # Only process pairs that are close enough to potentially violate
    close_mask = d_ij < 2.5 * R  # slightly wider margin for smooth transition
    if np.any(close_mask):
        i_close = i_idx[close_mask]
        j_close = j_idx[close_mask]
        d_close = d_ij[close_mask]
        
        # Smooth penalty: (max(0, 2R - d))^2
        violation = np.maximum(0.0, 2.0 * R - d_close)
        energy += np.sum(violation**2)
        
        # Gradient: d/dx_i sum (max(0, 2R - d_ij))^2
        # For pairs where 2R - d_ij > 0:
        # dE/dx_i = -2 * (2R - d_ij) * (x_i - x_j) / d_ij
        active = violation > 0
        if np.any(active):
            i_a = i_close[active]
            j_a = j_close[active]
            d_a = d_close[active]
            v_a = violation[active]
            
            factors = 2.0 * v_a / d_a
            vecs = centers[i_a] - centers[j_a]
            forces = factors[:, np.newaxis] * vecs
            
            np.add.at(grad, i_a, -forces)
            np.add.at(grad, j_a, forces)

    # 2. Wall terms - smooth penalty
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    
    wall_violation = np.maximum(0.0, R - wall_dists)
    energy += np.sum(wall_violation**2)
    
    active_wall = wall_violation > 0
    if np.any(active_wall):
        for idx in np.where(active_wall)[0]:
            v = wall_violation[idx]
            max_k = np.argmax(abs_centers[idx])
            sign_val = np.sign(centers[idx, max_k])
            if sign_val == 0:
                sign_val = 1.0
            grad[idx, max_k] += 2.0 * v * sign_val
            
    return energy, grad.flatten()
=======
def _energy_and_gradient(x, R):
    """
    Calculates the repulsion energy and gradient for a target radius R.
    Uses a smooth soft-penalty: quadratic when violated, zero otherwise.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise terms - smooth penalty for d_ij < 2R
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    dists = np.sqrt(sq_dists + 1e-15)
    np.fill_diagonal(dists, np.inf)
    np.fill_diagonal(sq_dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    d_ij = dists[i_idx, j_idx]
    
    # Only process pairs that are close enough to potentially violate
    close_mask = d_ij < 2.5 * R  # slightly wider margin for smooth transition
    if np.any(close_mask):
        i_close = i_idx[close_mask]
        j_close = j_idx[close_mask]
        d_close = d_ij[close_mask]
        
        # Smooth penalty: (max(0, 2R - d))^2
        violation = np.maximum(0.0, 2.0 * R - d_close)
        energy += np.sum(violation**2)
        
        # Gradient: d/dx_i sum (max(0, 2R - d_ij))^2
        # For pairs where 2R - d_ij > 0:
        # dE/dx_i = -2 * (2R - d_ij) * (x_i - x_j) / d_ij
        active = violation > 0
        if np.any(active):
            i_a = i_close[active]
            j_a = j_close[active]
            d_a = d_close[active]
            v_a = violation[active]
            
            factors = 2.0 * v_a / d_a
            vecs = centers[i_a] - centers[j_a]
            forces = factors[:, np.newaxis] * vecs
            
            np.add.at(grad, i_a, -forces)
            np.add.at(grad, j_a, forces)

    # 2. Wall terms - smooth penalty
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    
    wall_violation = np.maximum(0.0, R - wall_dists)
    energy += np.sum(wall_violation**2)
    
    active_wall = wall_violation > 0
    if np.any(active_wall):
        for idx in np.where(active_wall)[0]:
            v = wall_violation[idx]
            max_k = np.argmax(abs_centers[idx])
            sign_val = np.sign(centers[idx, max_k])
            if sign_val == 0:
                sign_val = 1.0
            grad[idx, max_k] += 2.0 * v * sign_val
            
    return energy, grad.flatten()


def _soft_min_objective_and_grad(x, temperature=0.01):
    """
    Soft-min surrogate for the minimum distance objective.
    Uses log-sum-exp to create a smooth approximation of min(d_ij/2, wall_dist_i).
    
    We want to maximize: min( min_{i<j} d_ij / 2, min_i wall_dist_i )
    
    This is equivalent to minimizing: -min(...)
    
    Using soft-min: -T * log(sum(exp(-d_k / T))) where d_k are all constraint distances
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    # Compute all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    dists = np.sqrt(sq_dists + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0  # radius constraint from pairs
    
    # Wall distances
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    
    # Combine all constraint "radii"
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Soft-min: T * log(sum(exp(-d/T)))
    # For numerical stability, subtract max
    min_d = np.min(all_dists)
    shifted = -all_dists / temperature
    shifted -= np.max(shifted)  # for stability
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = temperature * log_sum - min_d  # This gives approximately -min when T->0
    
    # Actually, let's use a cleaner formulation:
    # soft_min = -T * log(sum(exp(-d_k / T)))
    # As T -> 0, this approaches -min(d_k)
    # So we minimize soft_min (which is negative)
    
    exp_vals = np.exp(-all_dists / temperature)
    log_sum_exp = np.log(np.sum(exp_vals))
    soft_min = -temperature * log_sum_exp
    
    # Gradient of soft_min w.r.t. x
    # d(soft_min)/dx = -T * d(log(sum(exp(-d_k/T))))/dx
    # = -T * sum_k exp(-d_k/T) / sum_k exp(-d_k/T) * d(-d_k/T)/dx
    # = sum_k p_k * d_k_grad where p_k = exp(-d_k/T) / sum(exp(-d_k/T))
    
    probs = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(centers)
    
    # Pairwise gradient contributions
    # d(pair_dist_ij)/dx_i = (x_i - x_j) / d_ij
    pair_probs = probs[:len(pair_dists)]
    
    for k in range(len(pair_dists)):
        if pair_probs[k] < 1e-10:
            continue
        i, j = i_idx[k], j_idx[k]
        d_ij = dists[i, j]
        if d_ij < 1e-10:
            continue
        vec = (centers[i] - centers[j]) / d_ij
        grad[i] += pair_probs[k] * vec
        grad[j] -= pair_probs[k] * vec
    
    # Wall gradient contributions
    wall_probs = probs[len(pair_dists):]
    for i in range(n):
        if wall_probs[i] < 1e-10:
            continue
        max_k = np.argmax(np.abs(centers[i]))
        sign_val = np.sign(centers[i, max_k])
        if sign_val == 0:
            sign_val = 1.0
        # d(wall_dist_i)/dx_i = -sign for the max coordinate
        grad[i, max_k] -= wall_probs[i] * sign_val
    
    return soft_min, grad.flatten()
>>>>>>> REPLACE

