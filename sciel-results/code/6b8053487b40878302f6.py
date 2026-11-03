import numpy as np
import time
from scipy.optimize import least_squares

N = 300

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    # Precompute pair indices for efficiency
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    num_pairs = len(i_idx)
    
    def least_squares_residuals(x, target_r):
        """
        Residuals for least_squares: we want to maximize the minimum distance.
        We minimize the sum of squares of (target_r - d_i) for all constraints,
        where d_i are the half-distances between pairs and distances to boundary.
        This pushes all constraints to be at least target_r.
        """
        pts = x.reshape(N, 2)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        d_pair = np.sqrt(sq_dists[i_idx, j_idx]) / 2.0
        
        # Boundary distances
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        
        all_dists = np.concatenate([d_pair, bound_dists])
        
        # Residuals: push distances above target_r
        # If dist < target_r, residual is (target_r - dist) (positive)
        # If dist >= target_r, residual is 0
        residuals = np.maximum(0, target_r - all_dists)
        
        return residuals
    
    def least_squares_jacobian(x, target_r):
        """
        Analytical Jacobian for least_squares_residuals (vectorized).
        """
        pts = x.reshape(N, 2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        d_pair = np.sqrt(sq_dists[i_idx, j_idx]) / 2.0
        
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        
        all_dists = np.concatenate([d_pair, bound_dists])
        
        # Mask: only constraints where dist < target_r contribute
        mask = all_dists < target_r
        if not np.any(mask):
            return np.zeros((len(all_dists), N * 2))
        
        jac = np.zeros((len(all_dists), N * 2))
        
        # Pairwise constraints (vectorized)
        pair_mask = mask[:num_pairs]
        if np.any(pair_mask):
            idx_m = np.where(pair_mask)[0]
            d_pair_m = d_pair[idx_m]
            safe_d = np.where(d_pair_m > 1e-10, d_pair_m, 1e-10)
            
            pi = pts[i_idx[idx_m]]
            pj = pts[j_idx[idx_m]]
            vecs = (pi - pj) / (2.0 * safe_d[:, np.newaxis])
            
            ii = i_idx[idx_m]
            jj = j_idx[idx_m]
            
            # Vectorized assignment
            jac[idx_m, 2*ii] = -vecs[:, 0]
            jac[idx_m, 2*ii + 1] = -vecs[:, 1]
            jac[idx_m, 2*jj] = vecs[:, 0]
            jac[idx_m, 2*jj + 1] = vecs[:, 1]
        
        # Boundary constraints (vectorized)
        bound_mask = mask[num_pairs:]
        if np.any(bound_mask):
            idx_m = np.where(bound_mask)[0]
            rad_m = rad[idx_m]
            safe_rad = np.where(rad_m > 1e-10, rad_m, 1e-10)
            
            pi = pts[idx_m]
            vecs = pi / safe_rad[:, np.newaxis]
            
            rows = num_pairs + idx_m
            jac[rows, 2*idx_m] = vecs[:, 0]
            jac[rows, 2*idx_m + 1] = vecs[:, 1]
        
        return jac
    
    def run_least_squares(pts_init, target_r, time_budget, max_nfev=1000):
        """
        Run scipy least_squares to optimize packing with a target radius.
        Returns (best_pts, best_score, success).
        """
        x0 = pts_init.flatten()
        end_time = time.time() + time_budget
        
        def time_check(res):
            if time.time() > end_time:
                raise RuntimeError("Time limit exceeded")
        
        try:
            result = least_squares(
                least_squares_residuals,
                x0,
                args=(target_r,),
                jac=least_squares_jacobian,
                bounds=([-1.0, -1.0] * N, [1.0, 1.0] * N),
                method='trf',
                max_nfev=max_nfev,
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                callback=time_check
            )
            best_pts = result.x.reshape(N, 2)
            # Project back inside
            rad = np.linalg.norm(best_pts, axis=1)
            mask_out = rad > 1.0
            if np.any(mask_out):
                best_pts[mask_out] /= rad[mask_out, np.newaxis]
            
            best_score = compute_score(best_pts)
            return best_pts, best_score, True
        except RuntimeError:
            # Time limit hit, return current best if available
            return pts_init, compute_score(pts_init), False
        except Exception:
            return pts_init, compute_score(pts_init), False
    
    def compute_score(pts):
        """Compute the packing radius score efficiently."""
        # Compute squared distances to avoid sqrt where possible
        # Use vectorized computation
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        # Boundary distances
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        
        pair_vals = np.sqrt(sq_dists[i_idx, j_idx]) / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        return np.min(all_vals)
    
    def soft_min_weights(vals, k, current_min):
        """Compute soft-min weights using log-sum-exp for stability."""
        shifted = vals - current_min
        # Only consider constraints within a reasonable range
        mask_close = shifted < 30.0
        if not np.any(mask_close):
            return np.ones_like(vals) / len(vals)
        
        close_shifted = shifted[mask_close]
        # Use a shifted exponential to avoid overflow
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
        
        # For polish mode, use more aggressive settings
        if polish:
            stall_threshold = 200
            cooling = 0.9994
        
        while time.time() < end_time:
            # Compute distances - only compute what we need
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            sq_dists = np.sum(diff**2, axis=2)
            np.fill_diagonal(sq_dists, np.inf)
            
            rad = np.linalg.norm(pts, axis=1)
            bound_dists = 1.0 - rad
            
            # Only compute sqrt for pairs we need
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
            
            # Compute weights
            weights = soft_min_weights(all_vals, k, current_min)
            
            # Compute gradients
            grads = np.zeros_like(pts)
            
            # Pair contributions
            pair_weights = weights[:num_pairs]
            mask_nonzero = d_pair > 1e-10
            if np.any(mask_nonzero):
                # For each pair, gradient w.r.t. point i is (p_i - p_j)/|p_i - p_j| * weight
                # and w.r.t. point j is -(p_i - p_j)/|p_i - p_j| * weight
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
            
            # Adaptive step size based on progress
            if polish:
                # Smaller steps for polish
                step_alpha = alpha * 0.4
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
            
            # Increase k - use exponential schedule for better convergence
            progress = min(1.0, (time.time() - start_time) / time_budget)
            if polish:
                k = k_init + (k_end - k_init) * (progress ** 2.0)
            else:
                k = k_init + (k_end - k_init) * (progress ** 1.5)
            
            # Random restart on stall
            if stall_count > stall_threshold:
                noise_scale = noise_scale_base * (1.0 + alpha * 10)
                if polish:
                    noise_scale *= 0.2  # Even less noise during polish
                pts += np.random.normal(0, noise_scale, size=pts.shape)
                rad_new = np.linalg.norm(pts, axis=1)
                mask_out = rad_new > 1.0
                if np.any(mask_out):
                    pts[mask_out] /= rad_new[mask_out, np.newaxis]
                stall_count = 0
                alpha = max(alpha, 0.004 if not polish else 0.0015)
            
            iter_count += 1
        
        return best_pts, best_score
    
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
    # Focus on spacings near the optimal value
    spacing_multipliers = [0.95, 0.97, 0.99, 1.0, 1.01, 1.03]
    
    # Allocate time more evenly, with more time for better initializations
    time_per_init = min(15.0, time_limit * 0.18)
    
    for idx, mult in enumerate(spacing_multipliers):
        if time.time() - start_time > time_limit * 0.70:
            break
        
        spacing = base_spacing * mult
        pts = hex_lattice_init(spacing)
        
        # Run optimization with moderate settings - give more time to central spacings
        if 0.97 <= mult <= 1.03:
            time_budget = time_per_init
        else:
            time_budget = time_per_init * 0.6
        
        pts_opt, score = optimize(pts, 
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
    
    # Phase 2: Try random initializations - use fewer but better-tuned runs
    if time.time() - start_time < time_limit * 0.80:
        for _ in range(2):
            if time.time() - start_time > time_limit * 0.80:
                break
            
            pts = random_init()
            time_budget = min(12.0, time_limit * 0.12)
            pts_opt, score = optimize(pts,
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
    
    # Phase 3: Polish the best result found - use dedicated polish mode
    if best_pts is not None and time.time() - start_time < time_limit * 0.88:
        remaining_time = time_limit * 0.97 - time.time()
        if remaining_time > 5.0:
            # First polish pass with soft-min gradient ascent (shorter, more aggressive)
            pts_polished, score_polished = optimize(best_pts,
                                                     alpha_init=0.004,
                                                     k_init=150.0,
                                                     k_end=400.0,
                                                     cooling=0.9994,
                                                     time_budget=min(remaining_time * 0.25, 6.0),
                                                     stall_threshold=100,
                                                     noise_scale_base=0.002,
                                                     polish=True)
            
            if score_polished > best_score:
                best_score = score_polished
                best_pts = pts_polished
            
            # Second: Use scipy least_squares for high-precision polish with tighter target
            remaining_time2 = time_limit * 0.97 - time.time()
            if remaining_time2 > 4.0:
                # Try to push above current best_score with a slightly higher target
                target_r = best_score * 1.00005
                ls_budget = min(remaining_time2 * 0.45, 12.0)
                pts_ls, score_ls, success = run_least_squares(best_pts, target_r, ls_budget, max_nfev=2000)
                
                if score_ls > best_score:
                    best_score = score_ls
                    best_pts = pts_ls
            
            # Third: Another polish pass after least_squares to clean up
            remaining_time3 = time_limit * 0.97 - time.time()
            if remaining_time3 > 3.0:
                pts_polished2, score_polished2 = optimize(best_pts,
                                                           alpha_init=0.003,
                                                           k_init=200.0,
                                                           k_end=500.0,
                                                           cooling=0.9993,
                                                           time_budget=min(remaining_time3 * 0.3, 5.0),
                                                           stall_threshold=80,
                                                           noise_scale_base=0.001,
                                                           polish=True)
                
                if score_polished2 > best_score:
                    best_score = score_polished2
                    best_pts = pts_polished2
            
            # Fourth: Final high-precision least_squares
            remaining_time4 = time_limit * 0.97 - time.time()
            if remaining_time4 > 2.5:
                target_r4 = best_score * 1.00003
                ls_budget4 = min(remaining_time4 * 0.8, 8.0)
                pts_ls4, score_ls4, _ = run_least_squares(best_pts, target_r4, ls_budget4, max_nfev=1500)
                
                if score_ls4 > best_score:
                    best_score = score_ls4
                    best_pts = pts_ls4
    
    # Final verification
    if best_pts is None:
        # Fallback: simple hexagonal lattice
        best_pts = hex_lattice_init(base_spacing)
    
    # Ensure all points are inside the circle
    rad = np.linalg.norm(best_pts, axis=1)
    mask_out = rad > 1.0
    if np.any(mask_out):
        best_pts[mask_out] /= rad[mask_out, np.newaxis]
    
    # Final high-precision polish with scipy if time allows
    final_remaining = time_limit * 0.98 - time.time()
    if final_remaining > 2.0:
        target_r_final = best_score * 1.00002
        pts_final, score_final, _ = run_least_squares(best_pts, target_r_final, final_remaining, max_nfev=1000)
        if score_final > best_score:
            best_score = score_final
            best_pts = pts_final
    
    # Final safety check
    rad = np.linalg.norm(best_pts, axis=1)
    mask_out = rad > 1.0
    if np.any(mask_out):
        best_pts[mask_out] /= rad[mask_out, np.newaxis]
    
    return best_pts
