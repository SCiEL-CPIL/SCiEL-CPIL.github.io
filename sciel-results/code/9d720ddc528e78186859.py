import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    time_limit = 145.0
    
    # Precompute pair indices
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    num_pairs = len(i_idx)
    
    def _score(pts):
        """Compute the packing score."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists[i_idx, j_idx]) / 2.0
        rad = np.linalg.norm(pts, axis=1)
        bound_min = np.min(1.0 - rad)
        return min(pair_min, bound_min)
    
    def _soft_min_neg(pts, k=50.0):
        """Negative soft-min objective for minimization. Returns -soft_min(constraints)."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        pair_vals = dists[i_idx, j_idx] / 2.0
        rad = np.linalg.norm(pts, axis=1)
        bound_vals = 1.0 - rad
        
        all_vals = np.concatenate([pair_vals, bound_vals])
        
        # Soft-min: -log(sum(exp(-k*v))) / k
        # Use log-sum-exp trick for stability
        v_min = np.min(all_vals)
        shifted = all_vals - v_min
        # Only consider values close to min for numerical stability
        threshold = 50.0 / k
        mask = shifted < threshold
        if np.any(mask):
            close_shifted = shifted[mask]
            log_sum = np.log(np.sum(np.exp(-k * close_shifted)))
        else:
            log_sum = np.log(len(all_vals))
        
        soft_min_val = v_min - log_sum / k
        return -soft_min_val
    
    def _soft_min_grad(pts, k=50.0):
        """Gradient of the negative soft-min objective."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        pair_vals = dists[i_idx, j_idx] / 2.0
        rad = np.linalg.norm(pts, axis=1)
        bound_vals = 1.0 - rad
        
        all_vals = np.concatenate([pair_vals, bound_vals])
        v_min = np.min(all_vals)
        shifted = all_vals - v_min
        
        threshold = 50.0 / k
        mask = shifted < threshold
        if np.any(mask):
            close_shifted = shifted[mask]
            exp_terms = np.exp(-k * close_shifted)
            sum_exp = np.sum(exp_terms)
            if sum_exp > 0:
                weights = np.zeros_like(all_vals)
                weights[mask] = exp_terms / sum_exp
            else:
                weights = np.ones_like(all_vals) / len(all_vals)
        else:
            weights = np.ones_like(all_vals) / len(all_vals)
        
        grads = np.zeros_like(pts)
        
        # Pair gradients
        pair_weights = weights[:num_pairs]
        d_pair = dists[i_idx, j_idx]
        mask_nonzero = d_pair > 1e-9
        if np.any(mask_nonzero):
            vecs = np.zeros((num_pairs, 2))
            vecs[mask_nonzero] = (diff[i_idx[mask_nonzero], j_idx[mask_nonzero]] / d_pair[mask_nonzero, np.newaxis]) * pair_weights[mask_nonzero, np.newaxis]
            np.add.at(grads, i_idx, vecs)
            np.add.at(grads, j_idx, -vecs)
        
        # Boundary gradients
        bound_weights = weights[num_pairs:]
        rad_safe = np.where(rad > 1e-9, rad, 1.0)
        bound_grads = -(pts / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        return grads
    
    def _project_inside(pts):
        """Project points inside the unit circle."""
        rad = np.linalg.norm(pts, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            pts = pts.copy()
            pts[mask] /= rad[mask, np.newaxis]
        return pts
    
    # Generate hexagonal lattice initialization
    def _hex_init(spacing_factor=1.0):
        s_init = np.sqrt(2 * np.pi / (np.sqrt(3) * N)) * spacing_factor
        y_step = s_init * np.sqrt(3) / 2
        max_j = int(1.0 / y_step) + 2
        max_i = int(1.0 / s_init) + 2
        
        points_list = []
        for j in range(-max_j, max_j + 1):
            y = j * y_step
            offset = 0 if j % 2 == 0 else s_init / 2
            for i in range(-max_i, max_i + 1):
                x = i * s_init + offset
                if np.hypot(x, y) < 1.0:
                    points_list.append([x, y])
        
        points = np.array(points_list)
        if len(points) > N:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:N]
            points = points[idx]
        elif len(points) < N:
            while len(points) < N:
                p = np.random.uniform(-0.99, 0.99, 2)
                if np.linalg.norm(p) < 0.99:
                    points = np.vstack([points, p])
            points = points[:N]
        return points
    
    best_points = None
    best_score = -np.inf
    
    # Multiple restarts with different initializations and k values
    restart_configs = [
        (1.0, 30.0, 0.6),   # spacing factor, k, time fraction
        (0.95, 50.0, 0.3),
        (1.05, 20.0, 0.2),
        (0.9, 40.0, 0.15),
    ]
    
    time_remaining = time_limit - (time.time() - start_time)
    
    for spacing_factor, k_val, time_frac in restart_configs:
        if time.time() - start_time > time_limit * 0.85:
            break
        
        # Generate initial points
        points = _hex_init(spacing_factor)
        
        # Optimize using L-BFGS-B
        flat_pts = points.flatten()
        
        def objective(flat):
            pts = flat.reshape(N, 2)
            return _soft_min_neg(pts, k=k_val)
        
        def gradient(flat):
            pts = flat.reshape(N, 2)
            return _soft_min_grad(pts, k=k_val).flatten()
        
        # Run optimization
        try:
            result = minimize(
                objective,
                flat_pts,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 1000}
            )
        except Exception:
            continue
        
        optimized_pts = result.x.reshape(N, 2)
        optimized_pts = _project_inside(optimized_pts)
        
        score = _score(optimized_pts)
        if score > best_score:
            best_score = score
            best_points = optimized_pts.copy()
    
    # Final polish phase with very high k
    if best_points is not None and time.time() - start_time < time_limit * 0.9:
        points = best_points.copy()
        polish_k = 200.0
        polish_time_limit = time_limit * 0.95
        
        # Use scipy minimize for polish
        flat_pts = points.flatten()
        
        def polish_objective(flat):
            pts = flat.reshape(N, 2)
            return _soft_min_neg(pts, k=polish_k)
        
        def polish_gradient(flat):
            pts = flat.reshape(N, 2)
            return _soft_min_grad(pts, k=polish_k).flatten()
        
        try:
            result = minimize(
                polish_objective,
                flat_pts,
                jac=polish_gradient,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 500}
            )
            polished_pts = result.x.reshape(N, 2)
            polished_pts = _project_inside(polished_pts)
            
            polished_score = _score(polished_pts)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished_pts.copy()
        except Exception:
            pass
    
    # Final safety check
    if best_points is None:
        best_points = _hex_init(1.0)
    
    rad_final = np.linalg.norm(best_points, axis=1)
    mask_out = rad_final > 1.0
    if np.any(mask_out):
        best_points = best_points.copy()
        best_points[mask_out] /= rad_final[mask_out, np.newaxis]
    
    return best_points
