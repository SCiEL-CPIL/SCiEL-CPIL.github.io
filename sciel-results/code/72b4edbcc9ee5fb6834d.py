import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # 1. Initialization: Hexagonal Lattice
    s_init = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    s_gen = s_init * 0.95
    y_step = s_gen * np.sqrt(3) / 2
    max_j = int(1.0 / y_step) + 2
    max_i = int(1.0 / s_gen) + 2
    
    points_list = []
    for j in range(-max_j, max_j + 1):
        y = j * y_step
        offset = 0 if j % 2 == 0 else s_gen / 2
        for i in range(-max_i, max_i + 1):
            x = i * s_gen + offset
            if np.hypot(x, y) < 1.0:
                points_list.append([x, y])
                
    points = np.array(points_list)
    
    if len(points) > N:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        points = points[idx]
    elif len(points) < N:
        while len(points) < N:
            p = np.random.uniform(-1, 1, 2)
            if np.linalg.norm(p) < 1.0:
                points = np.vstack([points, p])
        points = points[:N]

    points = points.copy()
    best_points = points.copy()
    best_score = 0.0
    
    # Precompute indices for pairwise distances
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    num_pairs = len(i_idx)
    
    def _score(pts):
        """Compute the packing score."""
        d = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(d**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists[i_idx, j_idx]) / 2.0
        rad = np.linalg.norm(pts, axis=1)
        bound_min = np.min(1.0 - rad)
        return min(pair_min, bound_min)

    def _soft_min_value(vals, k_soft):
        """Compute the soft-min value."""
        current_min = np.min(vals)
        shifted = vals - current_min
        threshold = 50.0 / k_soft
        mask = shifted < threshold
        
        if np.any(mask):
            close_shifted = shifted[mask]
            z = -k_soft * close_shifted
            max_z = np.max(z)
            exp_z = np.exp(z - max_z)
            sum_exp = np.sum(exp_z)
            log_sum = max_z + np.log(sum_exp)
            return current_min - (1.0 / k_soft) * log_sum
        else:
            return current_min

    def _objective_and_grad(x_flat, k_soft, beta):
        """
        Objective: -soft_min(distances) + beta * barrier
        Returns value and gradient.
        """
        pts = x_flat.reshape(-1, 2)
        
        # Barrier
        rad = np.linalg.norm(pts, axis=1)
        rad_safe = np.minimum(rad, 0.999999)
        barrier_val = -np.sum(np.log(1.0 - rad_safe**2))
        barrier_grad = 2.0 * pts * (rad_safe**2 / (1.0 - rad_safe**2))
        
        # Distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        pair_dists = dists[i_idx, j_idx]
        bound_dists = 1.0 - rad_safe
        
        all_vals = np.concatenate([pair_dists, bound_dists])
        
        # Soft min value
        sm_val = _soft_min_value(all_vals, k_soft)
        
        # Soft min weights (for gradient)
        current_min = np.min(all_vals)
        shifted = all_vals - current_min
        threshold = 50.0 / k_soft
        mask = shifted < threshold
        
        weights = np.zeros_like(all_vals)
        if np.any(mask):
            close_shifted = shifted[mask]
            z = -k_soft * close_shifted
            max_z = np.max(z)
            exp_z = np.exp(z - max_z)
            sum_exp = np.sum(exp_z)
            weights[mask] = exp_z / sum_exp
        
        # Gradient of soft min w.r.t. positions
        grads = np.zeros_like(pts)
        
        # Pair gradients
        pair_weights = weights[:num_pairs]
        d_pair = pair_dists
        mask_nonzero = d_pair > 1e-9
        if np.any(mask_nonzero):
            vecs = np.zeros((num_pairs, 2))
            diff_pairs = diff[i_idx[mask_nonzero], j_idx[mask_nonzero]]
            vecs[mask_nonzero] = (diff_pairs / d_pair[mask_nonzero, np.newaxis]) * pair_weights[mask_nonzero, np.newaxis]
            
            np.add.at(grads, i_idx[mask_nonzero], vecs[mask_nonzero])
            np.add.at(grads, j_idx[mask_nonzero], -vecs[mask_nonzero])
            
        # Boundary gradients
        bound_weights = weights[num_pairs:]
        rad_safe_grad = np.where(rad_safe > 1e-9, rad_safe, 1.0)
        bound_grads = - (pts / rad_safe_grad[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        # Total gradient: grad(-softmin) + beta * grad(barrier)
        total_grad = grads + beta * barrier_grad
        
        val = -sm_val + beta * barrier_val
        
        return val, total_grad.flatten()

    # Optimization loop
    current_points = points.copy()
    
    k_soft = 10.0
    beta = 0.1
    
    for round_num in range(20):
        if time.time() - start_time > time_limit * 0.8:
            break
            
        res = minimize(
            _objective_and_grad,
            current_points.flatten(),
            args=(k_soft, beta),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res.success or res.status == 0:
            new_points = res.x.reshape(-1, 2)
            rad = np.linalg.norm(new_points, axis=1)
            mask = rad > 0.9999
            if np.any(mask):
                new_points[mask] *= 0.9999 / rad[mask, np.newaxis]
                
            score = _score(new_points)
            if score > best_score:
                best_score = score
                best_points = new_points.copy()
                current_points = new_points.copy()
                
        k_soft *= 2.0
        beta *= 1.5
        
        if k_soft > 1000:
             break

    # Final polish with the original gradient ascent for very fine tuning
    points = best_points.copy()
    polish_alpha = 0.001
    polish_k = 1000.0
    polish_time_limit = time_limit * 0.97
    
    def _soft_min_weights(vals, k_soft):
        current_min = np.min(vals)
        shifted = vals - current_min
        threshold = 50.0 / k_soft
        mask = shifted < threshold
        
        if np.any(mask):
            close_shifted = shifted[mask]
            z = -k_soft * close_shifted
            max_z = np.max(z)
            exp_z = np.exp(z - max_z)
            sum_exp = np.sum(exp_z)
            weights = np.zeros_like(vals)
            weights[mask] = exp_z / sum_exp
            return weights
        else:
            return np.ones_like(vals) / len(vals)

    while time.time() - start_time < polish_time_limit:
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        rad = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - rad
        
        pair_vals = dists[i_idx, j_idx] / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        current_min = np.min(all_vals)
        
        if current_min > best_score:
            best_score = current_min
            best_points = points.copy()
        
        weights = _soft_min_weights(all_vals, polish_k)
        
        grads = np.zeros_like(points)
        
        pair_weights = weights[:num_pairs]
        d_pair = dists[i_idx, j_idx]
        mask_nonzero = d_pair > 1e-9
        if np.any(mask_nonzero):
            vecs = np.zeros((num_pairs, 2))
            vecs[mask_nonzero] = (diff[i_idx[mask_nonzero], j_idx[mask_nonzero]] / d_pair[mask_nonzero, np.newaxis]) * pair_weights[mask_nonzero, np.newaxis]
            
            np.add.at(grads, i_idx, vecs)
            np.add.at(grads, j_idx, -vecs)
        
        bound_weights = weights[num_pairs:]
        rad_safe = np.where(rad > 1e-9, rad, 1.0)
        bound_grads = - (points / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm > 1.0:
            grads /= grad_norm
        
        points += polish_alpha * grads
        
        rad_new = np.linalg.norm(points, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            points[mask] /= rad_new[mask, np.newaxis]
        
        polish_alpha *= 0.99
        polish_k *= 1.05

    return best_points
