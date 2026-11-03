import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

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
    
    def get_active_constraints(pts, threshold=1e-4):
        """Identify constraints that are near the minimum (active)."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        
        d_pair = np.sqrt(sq_dists[i_idx, j_idx])
        pair_vals = d_pair / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        current_min = np.min(all_vals)
        active_mask = all_vals < (current_min + threshold)
        
        return active_mask, current_min
    
    def get_weights_and_grads(pts, k, current_min):
        """
        Compute soft-min weights and gradients stably using log-sum-exp.
        Returns: weights (for pairs and bounds), grads
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        
        d_pair = np.sqrt(sq_dists[i_idx, j_idx])
        pair_vals = d_pair / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        shifted = all_vals - current_min
        # Only consider constraints within a certain distance to focus on active ones
        max_shift = 10.0
        mask_active = shifted < max_shift
        
        weights = np.zeros_like(all_vals)
        if np.any(mask_active):
            active_vals = shifted[mask_active]
            
            # Use log-sum-exp for numerical stability
            # We want weights proportional to exp(-k * active_vals)
            # To avoid overflow/underflow, subtract the minimum value
            min_active = np.min(active_vals)
            shifted_active = active_vals - min_active  # Now min is 0
            
            # Compute log weights safely
            log_weights = -k * shifted_active
            max_log = np.max(log_weights)
            if np.isfinite(max_log):
                log_weights_shifted = log_weights - max_log
                exp_weights = np.exp(log_weights_shifted)
                sum_exp = np.sum(exp_weights)
                if sum_exp > 0:
                    weights_active = exp_weights / sum_exp
                    weights[mask_active] = weights_active
                else:
                    weights[mask_active] = 1.0 / np.sum(mask_active)
            else:
                weights[mask_active] = 1.0 / np.sum(mask_active)
        else:
            weights = np.ones_like(all_vals) / len(all_vals)
        
        grads = np.zeros_like(pts)
        
        # Pair contributions
        pair_weights = weights[:num_pairs]
        mask_nonzero = d_pair > 1e-9
        if np.any(mask_nonzero):
            idx_m = mask_nonzero
            # Vector direction from j to i
            vecs = diff[i_idx[idx_m], j_idx[idx_m]] / d_pair[idx_m, np.newaxis]
            # Weighted sum
            vecs *= pair_weights[idx_m, np.newaxis]
            
            np.add.at(grads, i_idx, vecs)
            np.add.at(grads, j_idx, -vecs)
        
        # Boundary contributions
        bound_weights = weights[num_pairs:]
        rad_safe = np.where(rad > 1e-9, rad, 1.0)
        # Gradient of (1 - |p|) wrt p is -p/|p|
        bound_grads = -(pts / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        return grads

    def optimize_soft_min(pts, alpha_init, k_init, k_end, cooling, time_budget, 
                         stall_threshold=150, noise_scale_base=0.01, polish=False):
        """
        Main optimization loop using soft-min gradient ascent.
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
            # Compute current min
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            sq_dists = np.sum(diff**2, axis=2)
            np.fill_diagonal(sq_dists, np.inf)
            
            rad = np.linalg.norm(pts, axis=1)
            bound_dists = 1.0 - rad
            
            d_pair_sq = sq_dists[i_idx, j_idx]
            d_pair = np.sqrt(d_pair_sq)
            
            pair_vals = d_pair / 2.0
            all_vals = np.concatenate([pair_vals, bound_dists])
            current_min = np.min(all_vals)
            
            if current_min > best_score:
                best_score = current_min
                best_pts = pts.copy()
                stall_count = 0
            else:
                stall_count += 1
            
            # Compute gradient using stable weights
            grads = get_weights_and_grads(pts, k, current_min)
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grads)
            if grad_norm > 1.0:
                grads /= grad_norm
            
            # Adaptive step size with line search
            if polish:
                step_alpha = alpha * 0.5
            else:
                step_alpha = alpha
            
            # Try the update
            new_pts = pts + step_alpha * grads
            
            # Project back inside circle
            rad_new = np.linalg.norm(new_pts, axis=1)
            mask_out = rad_new > 1.0
            if np.any(mask_out):
                new_pts[mask_out] /= rad_new[mask_out, np.newaxis]
            
            # Accept if it improves or slightly degrades (to avoid getting stuck)
            new_score = compute_score(new_pts)
            if new_score >= best_score - 1e-8:
                pts = new_pts
            else:
                # Try smaller step
                pts = pts + (step_alpha * 0.5) * grads
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
        
    def polish_direct(pts, time_budget):
        """
        High-precision polish using active constraint identification and targeted optimization.
        """
        current_pts = pts.copy()
        current_best = compute_score(pts)
        end_time = start_time + time_budget
        
        # Use high k for near-exact min
        k_final = 500.0
        alpha = 0.002
        cooling = 0.9998
        
        while time.time() < end_time:
            diff = current_pts[:, np.newaxis, :] - current_pts[np.newaxis, :, :]
            sq_dists = np.sum(diff**2, axis=2)
            np.fill_diagonal(sq_dists, np.inf)
            
            rad = np.linalg.norm(current_pts, axis=1)
            bound_dists = 1.0 - rad
            
            d_pair_sq = sq_dists[i_idx, j_idx]
            d_pair = np.sqrt(d_pair_sq)
            
            pair_vals = d_pair / 2.0
            all_vals = np.concatenate([pair_vals, bound_dists])
            current_min = np.min(all_vals)
            
            if current_min > current_best:
                current_best = current_min
            
            # Identify active constraints (within 1e-3 of minimum)
            active_mask, _ = get_active_constraints(current_pts, threshold=1e-3)
            
            # If we have few active constraints, use them for targeted optimization
            num_active = np.sum(active_mask)
            if num_active < 50:
                # Focus gradient on active constraints only
                grads = np.zeros_like(current_pts)
                
                # Pair contributions for active pairs
                pair_weights = active_mask[:num_pairs]
                mask_nonzero = (d_pair > 1e-9) & pair_weights
                if np.any(mask_nonzero):
                    idx_m = np.where(mask_nonzero)[0]
                    vecs = diff[i_idx[idx_m], j_idx[idx_m]] / d_pair[idx_m, np.newaxis]
                    vecs *= pair_weights[idx_m, np.newaxis]
                    np.add.at(grads, i_idx, vecs)
                    np.add.at(grads, j_idx, -vecs)
                
                # Boundary contributions for active bounds
                bound_weights = active_mask[num_pairs:]
                rad_safe = np.where(rad > 1e-9, rad, 1.0)
                bound_grads = -(current_pts / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
                grads += bound_grads
            else:
                # Use full soft-min gradient
                grads = get_weights_and_grads(current_pts, k_final, current_min)
            
            grad_norm = np.linalg.norm(grads)
            if grad_norm > 1e-12:
                grads /= grad_norm
                
                # Line search
                best_alpha = alpha
                best_score = current_min
                for factor in [1.0, 0.5, 0.25, 0.1]:
                    test_pts = current_pts + (alpha * factor) * grads
                    rad_test = np.linalg.norm(test_pts, axis=1)
                    mask_out = rad_test > 1.0
                    if np.any(mask_out):
                        test_pts[mask_out] /= rad_test[mask_out, np.newaxis]
                    
                    test_score = compute_score(test_pts)
                    if test_score > best_score:
                        best_score = test_score
                        best_alpha = alpha * factor
                        current_pts = test_pts
                    else:
                        break
                
                current_pts = current_pts + (best_alpha / alpha) * grads if best_alpha < alpha else current_pts
                rad_new = np.linalg.norm(current_pts, axis=1)
                mask_out = rad_new > 1.0
                if np.any(mask_out):
                    current_pts[mask_out] /= rad_new[mask_out, np.newaxis]
            
            alpha *= cooling
            if alpha < 1e-7:
                break
                
        return current_pts, current_best

    # Generate multiple initializations
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Try hexagonal lattice with various spacings
    base_spacing = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    spacing_multipliers = [0.95, 0.97, 0.99, 1.0, 1.01, 1.03]
    
    time_per_init = min(15.0, time_limit * 0.18)
    
    for idx, mult in enumerate(spacing_multipliers):
        if time.time() - start_time > time_limit * 0.65:
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
    if time.time() - start_time < time_limit * 0.75:
        for _ in range(2):
            if time.time() - start_time > time_limit * 0.75:
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
    if best_pts is not None and time.time() - start_time < time_limit * 0.85:
        remaining_time = time_limit * 0.97 - time.time()
        if remaining_time > 5.0:
            # First polish pass with soft-min gradient ascent
            pts_polished, score_polished = optimize_soft_min(best_pts,
                                                             alpha_init=0.005,
                                                             k_init=100.0,
                                                             k_end=300.0,
                                                             cooling=0.9995,
                                                             time_budget=min(remaining_time * 0.25, 6.0),
                                                             stall_threshold=150,
                                                             noise_scale_base=0.003,
                                                             polish=True)
            
            if score_polished > best_score:
                best_score = score_polished
                best_pts = pts_polished
            
            # Second: High-precision direct polish
            remaining_time2 = time_limit * 0.97 - time.time()
            if remaining_time2 > 3.0:
                pts_direct, score_direct = polish_direct(best_pts, min(remaining_time2 * 0.5, 10.0))
                
                if score_direct > best_score:
                    best_score = score_direct
                    best_pts = pts_direct
            
            # Third: Additional polish pass if time allows
            remaining_time3 = time_limit * 0.97 - time.time()
            if remaining_time3 > 2.0:
                pts_polish2, score_polish2 = polish_direct(best_pts, min(remaining_time3 * 0.6, 8.0))
                
                if score_polish2 > best_score:
                    best_score = score_polish2
                    best_pts = pts_polish2

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
        pts_final, score_final = polish_direct(best_pts, final_remaining)
        if score_final > best_score:
            best_score = score_final
            best_pts = pts_final
    
    # One more targeted polish focusing on the absolute minimum constraints
    final_remaining2 = time_limit * 0.99 - time.time()
    if final_remaining2 > 1.0:
        # Identify the single minimum constraint and try to improve it
        diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        rad = np.linalg.norm(best_pts, axis=1)
        bound_dists = 1.0 - rad
        
        d_pair_sq = sq_dists[i_idx, j_idx]
        d_pair = np.sqrt(d_pair_sq)
        
        pair_vals = d_pair / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        min_idx = np.argmin(all_vals)
        min_val = all_vals[min_idx]
        
        # If we have time, do a very targeted optimization
        if final_remaining2 > 0.5:
            # Small perturbation focused on improving the minimum
            noise = np.random.normal(0, 0.001, size=best_pts.shape)
            test_pts = best_pts + noise
            rad_test = np.linalg.norm(test_pts, axis=1)
            mask_out = rad_test > 1.0
            if np.any(mask_out):
                test_pts[mask_out] /= rad_test[mask_out, np.newaxis]
            
            test_score = compute_score(test_pts)
            if test_score > best_score:
                best_score = test_score
                best_pts = test_pts
    
    # Final safety check
    rad = np.linalg.norm(best_pts, axis=1)
    mask_out = rad > 1.0
    if np.any(mask_out):
        best_pts[mask_out] /= rad[mask_out, np.newaxis]
    
    return best_pts
