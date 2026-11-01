"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin for 150s limit
    
    def compute_radius(c):
        """Compute the actual radius for a configuration."""
        # c shape (N, 3)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        min_pair_dist = np.min(d[iu])
        
        # Wall distance
        # min_i min_k (0.5 - |c_ik|)
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    def soft_min_radius(c, beta=100.0):
        """Differentiable approximation of the radius."""
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        pair_dists = d[iu] / 2.0
        
        # Wall distances
        wall_dists = 0.5 - np.abs(c)
        wall_dists_flat = wall_dists.flatten()
        
        all_dists = np.concatenate([pair_dists, wall_dists_flat])
        
        # Soft min
        m = np.min(all_dists)
        # Avoid overflow in exp
        exp_terms = np.exp(-beta * (all_dists - m))
        soft_min_val = m - (1.0/beta) * np.log(np.sum(exp_terms))
        
        return -soft_min_val # We want to minimize negative radius

    def gradient_soft_min_radius(c, beta=100.0):
        """Gradient of the soft-min radius (negative for minimization)."""
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        pair_dists = d[iu] / 2.0
        
        # Wall distances
        wall_dists = 0.5 - np.abs(c)
        wall_dists_flat = wall_dists.flatten()
        
        all_dists = np.concatenate([pair_dists, wall_dists_flat])
        
        m = np.min(all_dists)
        exp_terms = np.exp(-beta * (all_dists - m))
        weights = exp_terms / np.sum(exp_terms)
        
        grad = np.zeros_like(c)
        
        # Gradient of pairwise part
        i_idx, j_idx = iu
        diff_iu = diff[i_idx, j_idx] # Shape (M, 3)
        d_iu = d[i_idx, j_idx]       # Shape (M,)
        
        # Avoid division by zero
        mask = d_iu > 1e-8
        if np.any(mask):
            unit_vec = np.zeros_like(diff_iu)
            unit_vec[mask] = diff_iu[mask] / d_iu[mask, np.newaxis]
            
            pair_weights = weights[:N*(N-1)//2]
            contrib = 0.5 * pair_weights[:, np.newaxis] * unit_vec
            
            # Use np.add.at for unbuffered addition (scatter-add)
            np.add.at(grad, i_idx, contrib)
            np.add.at(grad, j_idx, -contrib)
            
        # Gradient of wall part
        # d(softmin)/dc_ik = w_ik * d(0.5 - |c_ik|)/dc_ik = w_ik * (-sign(c_ik))
        wall_weights = weights[N*(N-1)//2:]
        wall_weights_reshaped = wall_weights.reshape((N, 3))
        signs = np.sign(c)
        
        # grad accumulates d(softmin)/dc
        grad -= wall_weights_reshaped * signs
        
        # We computed d(softmin)/dc.
        # We want to minimize -softmin, so we return -d(softmin)/dc.
        return -grad

    # 1. Initial Grid
    # 4x5x5 grid (100 points)
    x_ticks = (np.arange(4) + 0.5) / 4.0 - 0.5
    y_ticks = (np.arange(5) + 0.5) / 5.0 - 0.5
    z_ticks = (np.arange(5) + 0.5) / 5.0 - 0.5
    
    grid = np.array([(x, y, z) for z in z_ticks for y in y_ticks for x in x_ticks])
    c0 = grid[:N].copy()
    
    best_c = c0.copy()
    best_r = compute_radius(best_c)
    
    # 2. Gradient Ascent (Minimize -softmin)
    # Use simple gradient descent with adaptive step size
    current_c = c0.copy()
    step_size = 0.01
    prev_loss = soft_min_radius(current_c, beta=50.0)
    
    max_iter = 2000
    for i in range(max_iter):
        if time.time() - start_time > timeout:
            break
            
        grad = gradient_soft_min_radius(current_c, beta=50.0)
        
        # Check if gradient is zero
        if np.linalg.norm(grad) < 1e-8:
            break
            
        # Take step
        new_c = current_c - step_size * grad
        
        # Clip to box to keep valid? 
        # The soft-min objective handles walls, but keeping them inside helps stability
        new_c = np.clip(new_c, -0.499999, 0.499999)
        
        new_loss = soft_min_radius(new_c, beta=50.0)
        
        if new_loss < prev_loss:
            current_c = new_c
            prev_loss = new_loss
        else:
            # Decrease step
            step_size *= 0.5
            if step_size < 1e-5:
                break
                
        # Update best
        r_curr = compute_radius(current_c)
        if r_curr > best_r:
            best_r = r_curr
            best_c = current_c.copy()
            
    # 3. Local Polish with Nelder-Mead on exact objective
    # Flatten the best configuration
    x0 = best_c.flatten()
    
    def exact_obj(x):
        c = x.reshape((N, 3))
        # Ensure finite
        if not np.all(np.isfinite(c)):
            return 1e10
        # Check bounds strictly
        if np.any(c > 0.5) or np.any(c < -0.5):
            # Heavy penalty
            return 1e10 + np.sum(np.abs(c - np.clip(c, -0.5, 0.5)))
        
        r = compute_radius(c)
        return -r

    # Run Nelder-Mead
    # We need to be careful with time
    if time.time() - start_time < timeout - 10:
        try:
            res = minimize(exact_obj, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.success or res.fun < exact_obj(x0):
                c_nm = res.x.reshape((N, 3))
                # Validate
                if np.all(np.isfinite(c_nm)) and np.all(c_nm <= 0.5) and np.all(c_nm >= -0.5):
                    r_nm = compute_radius(c_nm)
                    if r_nm > best_r:
                        best_r = r_nm
                        best_c = c_nm
        except Exception:
            pass

    # Final check
    if time.time() - start_time > timeout:
         return best_c
         
    return best_c

# EVOLVE-BLOCK-END
