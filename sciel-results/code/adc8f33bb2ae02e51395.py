import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    time_limit = 143.0  # Safety margin before 150s kill
    
    # Precompute pair indices for efficiency
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    num_pairs = len(i_idx)
    
    def compute_score(pts):
        """Compute the packing radius score."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        
        pair_vals = dists[i_idx, j_idx] / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        return np.min(all_vals)
    
    def fast_distances(pts):
        """Compute pairwise distances and boundary distances efficiently."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        
        pair_vals = dists[i_idx, j_idx] / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        return dists, rad, bound_dists, all_vals
    
    def soft_min_weights(vals, k, current_min):
        """Compute soft-min weights using log-sum-exp for stability."""
        shifted = vals - current_min
        mask_close = shifted < 50.0
        if not np.any(mask_close):
            return np.ones_like(vals) / len(vals)
        
        close_shifted = shifted[mask_close]
        max_c = np.max(close_shifted)
        if max_c > 0:
            close_shifted = close_shifted - max_c
        exp_vals = np.exp(-k * close_shifted)
        weights = np.zeros_like(vals)
        weights[mask_close] = exp_vals
        sum_w = np.sum(weights)
        if sum_w > 1e-15:
            weights /= sum_w
        else:
            weights = np.ones_like(vals) / len(vals)
        return weights
    
    def optimize(pts, alpha_init, k_init, k_end, cooling, time_budget, 
                 stall_threshold=150, noise_scale_base=0.01):
        """
        Main optimization loop using soft-min gradient ascent.
        Returns best points found.
        """
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        alpha = alpha_init
        k = k_init
        stall_count = 0
        end_time = start_time + time_budget
        
        while time.time() < end_time:
            # Compute distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            rad = np.linalg.norm(pts, axis=1)
            bound_dists = 1.0 - rad
            
            pair_vals = dists[i_idx, j_idx] / 2.0
            all_vals = np.concatenate([pair_vals, bound_dists])
            current_min = np.min(all_vals)
            
            if current_min > best_score:
                best_score = current_min
                best_pts = pts.copy()
                stall_count = 0
            else:
                stall_count += 1
            
            # Compute weights
            weights = soft_min_weights(all_vals, k, current_min)
            
            # Compute gradients
            grads = np.zeros_like(pts)
            
            # Pair contributions
            pair_weights = weights[:num_pairs]
            d_pair = dists[i_idx, j_idx]
            mask_nonzero = d_pair > 1e-10
            if np.any(mask_nonzero):
                vecs = np.zeros((num_pairs, 2))
                idx_m = mask_nonzero
                vecs[idx_m] = (diff[i_idx[idx_m], j_idx[idx_m]] / 
                              d_pair[idx_m, np.newaxis]) * pair_weights[idx_m, np.newaxis]
                
                np.add.at(grads, i_idx, vecs)
                np.add.at(grads, j_idx, -vecs)
            
            # Boundary contributions
            bound_weights = weights[num_pairs:]
            rad_safe = np.where(rad > 1e-10, rad, 1.0)
            bound_grads = -(pts / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
            grads += bound_grads
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grads)
            if grad_norm > 1.0:
                grads /= grad_norm
            
            # Update
            pts += alpha * grads
            
            # Project back inside circle
            rad_new = np.linalg.norm(pts, axis=1)
            mask_out = rad_new > 1.0
            if np.any(mask_out):
                pts[mask_out] /= rad_new[mask_out, np.newaxis]
            
            # Cool down
            alpha *= cooling
            
            # Increase k
            progress = min(1.0, (time.time() - start_time) / time_budget)
            k = k_init + (k_end - k_init) * (progress ** 1.5)
            
            # Random restart on stall
            if stall_count > stall_threshold:
                noise_scale = noise_scale_base * (1.0 + alpha * 10)
                pts += np.random.normal(0, noise_scale, size=pts.shape)
                rad_new = np.linalg.norm(pts, axis=1)
                mask_out = rad_new > 1.0
                if np.any(mask_out):
                    pts[mask_out] /= rad_new[mask_out, np.newaxis]
                stall_count = 0
                alpha = max(alpha, 0.005)
        
        return best_pts, best_score
    
    def l_bfgs_b_polish(pts, time_budget=20.0):
        """
        Final polish using L-BFGS-B with the smooth surrogate objective.
        """
        if time.time() - start_time > time_limit - 5.0:
            return pts, compute_score(pts)
        
        def objective_and_grad(x):
            pts = x.reshape(-1, 2)
            
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            rad = np.linalg.norm(pts, axis=1)
            bound_dists = 1.0 - rad
            
            pair_vals = dists[i_idx, j_idx] / 2.0
            all_vals = np.concatenate([pair_vals, bound_dists])
            current_min = np.min(all_vals)
            
            # Use a smooth approximation: -log(sum(exp(-k*vals))) / k
            k = 300.0
            shifted = all_vals - current_min
            mask_close = shifted < 50.0
            if not np.any(mask_close):
                return -current_min, np.zeros_like(x)
            
            close_shifted = shifted[mask_close]
            max_c = np.max(close_shifted)
            if max_c > 0:
                close_shifted = close_shifted - max_c
            exp_vals = np.exp(-k * close_shifted)
            weights = np.zeros_like(all_vals)
            weights[mask_close] = exp_vals
            sum_w = np.sum(weights)
            if sum_w > 1e-15:
                weights /= sum_w
            
            # Compute gradient of the soft-min
            grads = np.zeros_like(pts)
            
            pair_weights = weights[:num_pairs]
            d_pair = dists[i_idx, j_idx]
            mask_nonzero = d_pair > 1e-10
            if np.any(mask_nonzero):
                vecs = np.zeros((num_pairs, 2))
                idx_m = mask_nonzero
                vecs[idx_m] = (diff[i_idx[idx_m], j_idx[idx_m]] / 
                              d_pair[idx_m, np.newaxis]) * pair_weights[idx_m, np.newaxis]
                
                np.add.at(grads, i_idx, vecs)
                np.add.at(grads, j_idx, -vecs)
            
            bound_weights = weights[num_pairs:]
            rad_safe = np.where(rad > 1e-10, rad, 1.0)
            bound_grads = -(pts / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
            grads += bound_grads
            
            # We want to maximize the soft-min, so minimize negative
            obj = -np.log(np.sum(exp_vals)) / k + current_min
            grad_flat = -grads.flatten()
            
            return obj, grad_flat
        
        x0 = pts.flatten()
        
        # Use L-BFGS-B with bounds to keep points inside circle
        # Note: L-BFGS-B doesn't handle non-convex constraints well,
        # so we'll use it as a local polish and verify afterward
        try:
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'maxls': 20, 'maxfun': 200,
                         'ftol': 1e-14, 'gtol': 1e-10, 'factr': 1e10}
            )
            
            if result.success or result.fun < objective_and_grad(x0)[0]:
                pts_new = result.x.reshape(-1, 2)
                # Project back inside circle
                rad_new = np.linalg.norm(pts_new, axis=1)
                mask_out = rad_new > 1.0
                if np.any(mask_out):
                    pts_new[mask_out] /= rad_new[mask_out, np.newaxis]
                
                score_new = compute_score(pts_new)
                score_old = compute_score(pts)
                
                if score_new >= score_old:
                    return pts_new, score_new
        except Exception:
            pass
        
        return pts, compute_score(pts)
    
    # Generate multiple initializations
    def hex_lattice_init(spacing):
        """Generate hexagonal lattice points inside unit circle."""
        y_step = spacing * np.sqrt(3) / 2
        max_j = int(1.0 / y_step) + 2
        max_i = int(1.0 / spacing) + 2
        
        pts = []
        for j in range(-max_j, max_j + 1):
            y = j * y_step
            offset = 0 if j % 2 == 0 else spacing / 2
            for i in range(-max_i, max_i + 1):
                x = i * spacing + offset
                if np.hypot(x, y) < 1.0:
                    pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) > N:
            dists = np.linalg.norm(pts, axis=1)
            idx = np.argsort(dists)[:N]
            pts = pts[idx]
        elif len(pts) < N:
            while len(pts) < N:
                p = np.random.uniform(-0.95, 0.95, 2)
                if np.linalg.norm(p) < 0.95:
                    pts = np.vstack([pts, p])
            pts = pts[:N]
        
        return pts
    
    def random_init():
        """Random initialization with rejection sampling."""
        pts = []
        while len(pts) < N:
            p = np.random.uniform(-0.98, 0.98, 2)
            if np.linalg.norm(p) < 0.98:
                pts.append(p)
        return np.array(pts)
    
    # Try multiple initializations and optimization settings
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Try hexagonal lattice with various spacings
    base_spacing = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    spacing_multipliers = [0.90, 0.93, 0.96, 0.99, 1.02, 1.05]
    
    for mult in spacing_multipliers:
        if time.time() - start_time > time_limit * 0.65:
            break
        
        spacing = base_spacing * mult
        pts = hex_lattice_init(spacing)
        
        # Run optimization with moderate settings
        time_budget = min(12.0, time_limit * 0.13)
        pts_opt, score = optimize(pts, 
                                   alpha_init=0.012, 
                                   k_init=2.0, 
                                   k_end=200.0, 
                                   cooling=0.9999,
                                   time_budget=time_budget,
                                   stall_threshold=150,
                                   noise_scale_base=0.006)
        
        if score > best_score:
            best_score = score
            best_pts = pts_opt.copy()
    
    # Phase 2: Try random initializations
    if time.time() - start_time < time_limit * 0.75:
        for _ in range(4):
            if time.time() - start_time > time_limit * 0.75:
                break
            
            pts = random_init()
            time_budget = min(10.0, time_limit * 0.10)
            pts_opt, score = optimize(pts,
                                       alpha_init=0.008,
                                       k_init=1.5,
                                       k_end=150.0,
                                       cooling=0.9998,
                                       time_budget=time_budget,
                                       stall_threshold=100,
                                       noise_scale_base=0.008)
            
            if score > best_score:
                best_score = score
                best_pts = pts_opt.copy()
    
    # Phase 3: Polish the best result found with gradient-based optimizer
    if best_pts is not None and time.time() - start_time < time_limit * 0.85:
        remaining_time = time_limit * 0.92 - time.time()
        if remaining_time > 3.0:
            pts_polished, score_polished = optimize(best_pts,
                                                     alpha_init=0.002,
                                                     k_init=300.0,
                                                     k_end=800.0,
                                                     cooling=0.9995,
                                                     time_budget=min(remaining_time * 0.4, 8.0),
                                                     stall_threshold=300,
                                                     noise_scale_base=0.001)
            
            if score_polished > best_score:
                best_score = score_polished
                best_pts = pts_polished
    
    # Phase 4: Final L-BFGS-B polish
    if best_pts is not None and time.time() - start_time < time_limit * 0.93:
        remaining_time = time_limit * 0.97 - time.time()
        if remaining_time > 5.0:
            pts_final, score_final = l_bfgs_b_polish(best_pts, time_budget=min(remaining_time, 20.0))
            
            if score_final > best_score:
                best_score = score_final
                best_pts = pts_final
    
    # Phase 5: Ultra-precise final polish with very high k
    if best_pts is not None and time.time() - start_time < time_limit * 0.96:
        remaining_time = time_limit * 0.985 - time.time()
        if remaining_time > 3.0:
            pts_ultra, score_ultra = optimize(best_pts,
                                               alpha_init=0.001,
                                               k_init=500.0,
                                               k_end=2000.0,
                                               cooling=0.9998,
                                               time_budget=min(remaining_time * 0.5, 5.0),
                                               stall_threshold=500,
                                               noise_scale_base=0.0005)
            
            if score_ultra > best_score:
                best_score = score_ultra
                best_pts = pts_ultra
    
    # Final verification
    if best_pts is None:
        # Fallback: simple hexagonal lattice
        best_pts = hex_lattice_init(base_spacing)
    
    # Ensure all points are inside the circle
    rad = np.linalg.norm(best_pts, axis=1)
    mask_out = rad > 1.0
    if np.any(mask_out):
        best_pts[mask_out] /= rad[mask_out, np.newaxis]
    
    return best_pts
