import numpy as np

def construct():
    N = 50
    # Parameters: R, theta[0..N-1], phi[0..N-1]
    # Initialize R
    R = 0.35
    # Initialize angles using Fibonacci sphere
    thetas = np.zeros(N)
    phis = np.zeros(N)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(N):
        cos_theta = 1.0 - 2.0 * (i + 0.5) / N
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        phi = i * golden_angle
        # Convert to theta, phi in spherical coords (theta from z-axis)
        thetas[i] = np.arccos(cos_theta)
        phis[i] = phi

    max_iter = 20000
    step_size = 0.01
    best_r = 0.0
    best_params = (R, thetas.copy(), phis.copy())

    for i in range(max_iter):
        # Calculate centers
        sin_thetas = np.sin(thetas)
        cos_thetas = np.cos(thetas)
        cos_phis = np.cos(phis)
        sin_phis = np.sin(phis)
        
        x = R * sin_thetas * cos_phis
        y = R * sin_thetas * sin_phis
        z = R * cos_thetas
        
        centers = np.column_stack((x, y, z))
        
        # Calculate distances and wall clearances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        abs_centers = np.abs(centers)
        wall_dists = 0.5 - abs_centers
        # min wall dist per sphere is min over 3 coords
        min_wall_per_sphere = np.min(wall_dists, axis=1)
        
        min_pair_dist = np.min(dist[~np.eye(N, dtype=bool)])
        min_wall_dist = np.min(min_wall_per_sphere)
        
        current_r = min(min_pair_dist / 2.0, min_wall_dist)
        
        if current_r > best_r:
            best_r = current_r
            best_params = (R, thetas.copy(), phis.copy())
            
        # Gradients
        # We want to maximize r.
        # Let's identify active constraints.
        # Pair constraints: D_ij / 2 <= r
        # Wall constraints: W_ik <= r
        
        # Threshold for active
        delta = 1e-3
        active_pairs = []
        active_walls = [] # (i, k)
        
        # Find active pairs
        # dist is NxN
        # We need D_ij < 2*(current_r + delta)
        active_pair_mask = dist < 2.0 * (current_r + delta)
        np.fill_diagonal(active_pair_mask, False)
        active_pairs = np.argwhere(active_pair_mask)
        # Ensure i < j
        active_pairs = active_pairs[active_pairs[:, 0] < active_pairs[:, 1]]
        
        # Find active walls
        # wall_dists is Nx3
        active_wall_mask = wall_dists < current_r + delta
        active_walls = np.argwhere(active_wall_mask)
        
        # Gradient of r w.r.t R, thetas, phis
        # dr/dR, dr/dtheta_i, dr/dphi_i
        grad_R = 0.0
        grad_thetas = np.zeros(N)
        grad_phis = np.zeros(N)
        
        # Pair contributions
        # d(D_ij)/dR = 2 * (c_i - c_j) . (dc_i/dR) ... 
        # c_i = R * s_i, where s_i is unit vector.
        # dc_i/dR = s_i
        # d(D_ij)/dR = (s_i - s_j) . (c_i - c_j) / D_ij * 2 ? No.
        # D_ij = |c_i - c_j|. 
        # dD_ij/dR = (c_i - c_j) . (s_i - s_j) / D_ij
        # dr/dR from pair = 0.5 * dD_ij/dR
        
        # d(D_ij)/dtheta_k = (c_i - c_j) . (dc_i/dtheta_k - dc_j/dtheta_k) / D_ij
        # dc/dtheta = R * (cos(theta)cos(phi), cos(theta)sin(phi), -sin(theta))
        
        # d(D_ij)/dphi_k = (c_i - c_j) . (dc_i/dphi_k - dc_j/dphi_k) / D_ij
        # dc/dphi = R * (-sin(theta)sin(phi), sin(theta)cos(phi), 0)
        
        # Wall contributions
        # W_ik = 0.5 - |c_ik|
        # dW_ik/dR = -sign(c_ik) * d(c_ik)/dR = -sign(c_ik) * s_ik
        # dW_ik/dtheta_j = -sign(c_ik) * d(c_ik)/dtheta_j (if i==j else 0)
        # dW_ik/dphi_j = -sign(c_ik) * d(c_ik)/dphi_j (if i==j else 0)
        
        # Precompute derivatives of unit vectors
        # s_i = (sin(t)c(p), sin(t)s(p), cos(t))
        # ds_i/dt = (cos(t)c(p), cos(t)s(p), -sin(t))
        # ds_i/dp = (-sin(t)s(p), sin(t)c(p), 0)
        
        ds_dthetas = np.column_stack((
            cos_thetas * cos_phis,
            cos_thetas * sin_phis,
            -sin_thetas
        ))
        ds_dphis = np.column_stack((
            -sin_thetas * sin_phis,
            sin_thetas * cos_phis,
            np.zeros(N)
        ))
        
        # Accumulate gradients
        for i_idx, j_idx in active_pairs:
            c_i = centers[i_idx]
            c_j = centers[j_idx]
            d_ij = dist[i_idx, j_idx]
            if d_ij < 1e-9: continue
            
            v = c_i - c_j
            v_norm = v / d_ij
            
            # dD_ij/dR = v_norm . (s_i - s_j)
            # s_i is unit vector = c_i / R
            s_i = c_i / R
            s_j = c_j / R
            dD_dR = np.dot(v_norm, s_i - s_j)
            grad_R += 0.5 * dD_dR
            
            # dD_ij/dtheta_k = v_norm . (R * (ds_i/dtheta_k - ds_j/dtheta_k))
            # Only non-zero for k=i and k=j
            # For k=i: R * ds_i/dtheta_i
            # For k=j: -R * ds_j/dtheta_j
            grad_thetas[i_idx] += 0.5 * np.dot(v_norm, R * ds_dthetas[i_idx])
            grad_thetas[j_idx] -= 0.5 * np.dot(v_norm, R * ds_dthetas[j_idx])
            
            grad_phis[i_idx] += 0.5 * np.dot(v_norm, R * ds_dphis[i_idx])
            grad_phis[j_idx] -= 0.5 * np.dot(v_norm, R * ds_dphis[j_idx])
            
        for i_idx, k_idx in active_walls:
            c_ik = centers[i_idx, k_idx]
            if abs(c_ik) < 1e-9:
                # Gradient undefined, skip or handle carefully
                continue
            sign_c = np.sign(c_ik)
            
            # dW_ik/dR = -sign_c * s_ik
            s_ik = centers[i_idx, k_idx] / R
            grad_R += -sign_c * s_ik
            
            # dW_ik/dtheta_i = -sign_c * R * (ds_i/dtheta)_k
            # (ds_i/dtheta)_k is the k-th component of ds_dthetas[i_idx]
            grad_thetas[i_idx] += -sign_c * R * ds_dthetas[i_idx, k_idx]
            
            # dW_ik/dphi_i = -sign_c * R * (ds_i/dphi)_k
            grad_phis[i_idx] += -sign_c * R * ds_dphis[i_idx, k_idx]
            
        # Normalize gradients to prevent large steps if many active constraints
        norm_grad_R = abs(grad_R)
        norm_grad_thetas = np.linalg.norm(grad_thetas)
        norm_grad_phis = np.linalg.norm(grad_phis)
        
        if norm_grad_R > 1e-9:
            grad_R /= norm_grad_R
        if norm_grad_thetas > 1e-9:
            grad_thetas /= norm_grad_thetas
        if norm_grad_phis > 1e-9:
            grad_phis /= norm_grad_phis
            
        # Update
        R += step_size * grad_R
        thetas += step_size * grad_thetas
        phis += step_size * grad_phis
        
        # Keep R positive and reasonable
        R = max(R, 0.01)
        R = min(R, 0.49) # Must be less than 0.5 to have wall clearance
        
        # Decay
        step_size *= 0.999
        
    # Reconstruct best centers
    R_best, thetas_best, phis_best = best_params
    sin_thetas_best = np.sin(thetas_best)
    cos_thetas_best = np.cos(thetas_best)
    cos_phis_best = np.cos(phis_best)
    sin_phis_best = np.sin(phis_best)
    
    x_best = R_best * sin_thetas_best * cos_phis_best
    y_best = R_best * sin_thetas_best * sin_phis_best
    z_best = R_best * cos_thetas_best
    
    centers_best = np.column_stack((x_best, y_best, z_best))
    return centers_best
