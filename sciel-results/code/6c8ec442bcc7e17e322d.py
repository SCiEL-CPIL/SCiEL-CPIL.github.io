def _energy_and_gradient_vec(x, R):
    centers = x.reshape(-1, 3)
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    mask = dists < 2.0 * R
    # Avoid dividing by zero or inf
    inv_dists = np.zeros_like(dists)
    inv_dists[mask] = 1.0 / dists[mask]
    
    # v_ij = 2R - d_ij for masked pairs
    v = np.zeros_like(dists)
    v[mask] = 2.0 * R - dists[mask]
    
    # Force on i from j: -2 * v_ij * (c_i - c_j) / d_ij
    # Note: diff[i,j] = c_i - c_j
    # force_ij = -2 * v_ij * diff[i,j] / d_ij
    forces = np.zeros_like(diff)
    forces[mask] = -2.0 * v[mask, :, np.newaxis] * diff[mask, :, np.newaxis] * inv_dists[mask, :, np.newaxis]
    
    # Sum forces over neighbors for each center
    # grad[i] = sum_j force_ij
    grad = np.sum(forces, axis=1)
    
    energy = np.sum(v[mask]**2)
    
    # Wall terms
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    mask_wall = wall_dists < R
    v_wall = np.zeros(n)
    v_wall[mask_wall] = R - wall_dists[mask_wall]
    
    energy += np.sum(v_wall**2)
    
    # Wall gradient
    grad_wall = np.zeros_like(centers)
    for i in range(n):
        if mask_wall[i]:
            v = v_wall[i]
            max_val = max_abs[i]
            for k in range(3):
                if np.isclose(abs_centers[i, k], max_val):
                    grad_wall[i, k] = 2.0 * v * np.sign(centers[i, k])
                    break
    grad += grad_wall
    
    return energy, grad.flatten()
