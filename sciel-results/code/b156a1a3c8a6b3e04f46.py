import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    # Precompute pair indices for efficiency
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    num_pairs = len(i_idx)
    
    def compute_score(pts):
        """Compute the packing radius score efficiently."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        
        pair_vals = np.sqrt(sq_dists[i_idx, j_idx]) / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        return np.min(all_vals)
    
    def compute_distances(pts):
        """Compute all pairwise distances and boundary distances."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        
        d_pair = np.sqrt(sq_dists[i_idx, j_idx])
        
        return diff, d_pair, rad, bound_dists
    
    def soft_min_objective(pts, k, current_min):
        """
        Soft-min surrogate: -log(sum(exp(-k*(d_i - current_min)))) / k
        This is a smooth approximation of min(d_i).
        """
        diff, d_pair, rad, bound_dists = compute_distances(pts)
        
        pair_vals = d_pair / 2.0
        all_dists = np.concatenate([pair_vals, bound_dists])
        
        shifted = all_dists - current_min
        shifted = np.clip(shifted, -100, 100)
        exp_vals = np.exp(-k * shifted)
        log_sum = np.log(np.sum(exp_vals))
        return -log_sum / k
    
    def soft_min_gradient(pts, k, current_min):
        """
        Gradient of the soft-min objective.
        """
        diff, d_pair, rad, bound_dists = compute_distances(pts)
        
        pair_vals = d_pair / 2.0
        all_dists = np.concatenate([pair_vals, bound_dists])
        
        shifted = all_dists - current_min
        shifted = np.clip(shifted, -100, 100)
        exp_vals = np.exp(-k * shifted)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp
        
        grads = np.zeros_like(pts)
        
        # Pair contributions
        pair_weights = weights[:num_pairs]
        mask_nonzero = d_pair > 1e-10
        if np.any(mask_nonzero):
            idx_m = mask_nonzero
            # Use the precomputed diff for efficiency
            vecs = (diff[i_idx[idx_m], j_idx[idx_m]] / 
                   d_pair[idx_m, np.newaxis]) * pair_weights[idx_m, np.newaxis]
            
            np.add.at(grads, i_idx, vecs)
            np.add.at(grads, j_idx, -vecs)
        
        # Boundary contributions
        bound_weights = weights[num_pairs:]
        rad_safe = np.where(rad > 1e-10, rad, 1.0)
        bound_grads = -(pts / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        return grads
    
    def optimize_soft_min(pts, alpha_init, k_init, k_end, cooling, time_budget, 
                         stall_threshold=150, noise_scale_base=0.01, polish=False):
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
        iter_count = 0
        end_time = start_time + time_budget
        
        if polish:
            stall_threshold = 300
            cooling = 0.9995
        
        while time.time() < end_time:
            # Compute distances once
            diff, d_pair, rad, bound_dists = compute_distances(pts)
            
            pair_vals = d_pair / 2.0
            all_vals = np.concatenate([pair_vals, bound_dists])
            current_min = np.min(all_vals)
            
            if current_min > best_score:
                best_score = current_min
                best_pts = pts.copy()
                stall_count = 0
            else:
                stall_count += 1
            
            # Compute gradient of soft-min (reuses distance computation via closure)
            grads = soft_min_gradient(pts, k, current_min)
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grads)
            if grad_norm > 1.0:
                grads /= grad_norm
            
            # Adaptive step size
            if polish:
                step_alpha = alpha * 0.5
            else:
                step_alpha = alpha
            
            # Update
            pts += step_alpha * grads
            
            # Project back inside circle
            rad_new = np.linalg.norm(pts, axis=1)
            mask_out = rad_new > 1.0
            if np.any(mask_out):
                pts[mask_out] /= rad_new[mask_out, np.newaxis]
            
            # Cool down
            alpha *= cooling
            
            # Increase k
            progress = min(1.0, (time.time() - start_time) / time_budget)
            if polish:
                k = k_init + (k_end - k_init) * (progress ** 2.0)
            else:
                k = k_init + (k_end - k_init) * (progress ** 1.5)
            
            # Random restart on stall
            if stall_count > stall_threshold:
                noise_scale = noise_scale_base * (1.0 + alpha * 10)
                if polish:
                    noise_scale *= 0.3
                pts += np.random.normal(0, noise_scale, size=pts.shape)
                rad_new = np.linalg.norm(pts, axis=1)
                mask_out = rad_new > 1.0
                if np.any(mask_out):
                    pts[mask_out] /= rad_new[mask_out, np.newaxis]
                stall_count = 0
                alpha = max(alpha, 0.005 if not polish else 0.002)
            
            iter_count += 1
        
        return best_pts, best_score
    
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
    
    def polish_with_scipy(pts, time_budget, k_start=200.0, k_end=5000.0):
        """
        Use scipy minimize to polish the packing.
        Multiple passes with increasing k for near-exact min.
        """
        end_time = start_time + time_budget
        current_best = compute_score(pts)
        current_pts = pts.copy()
        
        # Multiple k values for progressive refinement
        k_values = [k_start, k_start * 3, k_start * 10, k_end]
        
        for k_val in k_values:
            if time.time() > end_time:
                break
            
            def objective_k(x):
                pts_x = x.reshape(N, 2)
                return soft_min_objective(pts_x, k_val, current_best)
            
            def gradient_k(x):
                pts_x = x.reshape(N, 2)
                return soft_min_gradient(pts_x, k_val, current_best).flatten()
            
            x0 = current_pts.flatten()
            
            try:
                result = minimize(
                    objective_k,
                    x0,
                    jac=gradient_k,
                    method='L-BFGS-B',
                    bounds=[(-1.0, 1.0)] * (2 * N),
                    options={'maxiter': 150, 'ftol': 1e-16, 'gtol': 1e-16, 'maxls': 50}
                )
                pts_result = result.x.reshape(N, 2)
                
                # Project back inside
                rad = np.linalg.norm(pts_result, axis=1)
                mask_out = rad > 1.0
                if np.any(mask_out):
                    pts_result[mask_out] /= rad[mask_out, np.newaxis]
                
                score_result = compute_score(pts_result)
                if score_result > current_best:
                    current_best = score_result
                    current_pts = pts_result
            except Exception:
                pass
        
        return current_pts, current_best
    
    def polish_with_high_k(pts, time_budget):
        """
        Ultra-high precision polish using very high k values.
        Multiple passes with increasing k.
        """
        return polish_with_scipy(pts, time_budget, k_start=500.0, k_end=10000.0)
    
    # Generate multiple initializations
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Try hexagonal lattice with various spacings
    base_spacing = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    spacing_multipliers = [0.95, 0.97, 0.99, 1.0, 1.01, 1.03]
    
    time_per_init = min(15.0, time_limit * 0.18)
    
    for idx, mult in enumerate(spacing_multipliers):
        if time.time() - start_time > time_limit * 0.60:
            break
        
        spacing = base_spacing * mult
        pts = hex_lattice_init(spacing)
        
        if 0.97 <= mult <= 1.03:
            time_budget = time_per_init
        else:
            time_budget = time_per_init * 0.6
        
        pts_opt, score = optimize_soft_min(pts, 
                                           alpha_init=0.012, 
                                           k_init=5.0, 
                                           k_end=200.0, 
                                           cooling=0.9997,
                                           time_budget=time_budget,
                                           stall_threshold=120,
                                           noise_scale_base=0.006)
        
        if score > best_score:
            best_score = score
            best_pts = pts_opt.copy()
    
    # Phase 2: Try random initializations
    if time.time() - start_time < time_limit * 0.70:
        for _ in range(2):
            if time.time() - start_time > time_limit * 0.70:
                break
            
            pts = random_init()
            time_budget = min(12.0, time_limit * 0.12)
            pts_opt, score = optimize_soft_min(pts,
                                               alpha_init=0.008,
                                               k_init=4.0,
                                               k_end=150.0,
                                               cooling=0.9996,
                                               time_budget=time_budget,
                                               stall_threshold=100,
                                               noise_scale_base=0.008)
            
            if score > best_score:
                best_score = score
                best_pts = pts_opt.copy()
    
    # Phase 3: Polish the best result
    if best_pts is not None and time.time() - start_time < time_limit * 0.80:
        remaining_time = time_limit * 0.97 - time.time()
        if remaining_time > 5.0:
            # First polish pass with soft-min gradient ascent
            pts_polished, score_polished = optimize_soft_min(best_pts,
                                                             alpha_init=0.005,
                                                             k_init=100.0,
                                                             k_end=300.0,
                                                             cooling=0.9995,
                                                             time_budget=min(remaining_time * 0.20, 5.0),
                                                             stall_threshold=150,
                                                             noise_scale_base=0.003,
                                                             polish=True)
            
            if score_polished > best_score:
                best_score = score_polished
                best_pts = pts_polished
            
            # Second: Use scipy for high-precision polish with progressive k
            remaining_time2 = time_limit * 0.97 - time.time()
            if remaining_time2 > 3.0:
                pts_scipy, score_scipy = polish_with_scipy(best_pts, min(remaining_time2 * 0.5, 15.0))
                
                if score_scipy > best_score:
                    best_score = score_scipy
                    best_pts = pts_scipy
            
            # Third: Ultra-high precision polish
            remaining_time3 = time_limit * 0.97 - time.time()
            if remaining_time3 > 3.0:
                pts_ultra, score_ultra = polish_with_high_k(best_pts, min(remaining_time3 * 0.9, 25.0))
                
                if score_ultra > best_score:
                    best_score = score_ultra
                    best_pts = pts_ultra
    
    # Final verification
    if best_pts is None:
        best_pts = hex_lattice_init(base_spacing)
    
    # Ensure all points are inside the circle
    rad = np.linalg.norm(best_pts, axis=1)
    mask_out = rad > 1.0
    if np.any(mask_out):
        best_pts[mask_out] /= rad[mask_out, np.newaxis]
    
    # Final high-precision polish if time allows
    final_remaining = time_limit * 0.98 - time.time()
    if final_remaining > 2.0:
        pts_final, score_final = polish_with_high_k(best_pts, final_remaining)
        if score_final > best_score:
            best_score = score_final
            best_pts = pts_final
    
    # Final safety check
    rad = np.linalg.norm(best_pts, axis=1)
    mask_out = rad > 1.0
    if np.any(mask_out):
        best_pts[mask_out] /= rad[mask_out, np.newaxis]
    
    return best_pts
