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
    
    def soft_min_objective_and_grad(pts, beta, offset=0.0):
        """
        Compute the soft-min objective and its gradient.
        Objective: -log(sum(exp(-beta*(d_i - offset)))) / beta
        Gradient: sum_j [ w_j * grad(d_j) ]
        where w_j = exp(-beta*(d_j - offset)) / sum(exp(-beta*(d_i - offset)))
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        
        d_pair = np.sqrt(sq_dists[i_idx, j_idx]) / 2.0
        all_dists = np.concatenate([d_pair, bound_dists])
        
        # Shift to avoid underflow/overflow
        shifted = all_dists - offset
        shifted = np.clip(shifted, -50, 50)
        
        exp_vals = np.exp(-beta * shifted)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp
        
        obj = -np.log(sum_exp) / beta + offset
        
        grads = np.zeros_like(pts)
        
        # Pair contributions
        pair_weights = weights[:num_pairs]
        mask_nonzero = sq_dists[i_idx, j_idx] > 1e-10
        if np.any(mask_nonzero):
            idx_m = mask_nonzero
            # d_pair = 0.5 * dist
            # grad(d_pair) for i: 0.5 * (pts[i] - pts[j]) / dist
            # grad(d_pair) for j: 0.5 * (pts[j] - pts[i]) / dist
            dists = np.sqrt(sq_dists[i_idx[idx_m], j_idx[idx_m]])
            vecs = diff[i_idx[idx_m], j_idx[idx_m]] / dists[:, np.newaxis]
            # weight * 0.5 * vec
            contribs = vecs * (0.5 * pair_weights[idx_m])[:, np.newaxis]
            
            np.add.at(grads, i_idx[idx_m], contribs)
            np.add.at(grads, j_idx[idx_m], -contribs)
        
        # Boundary contributions
        bound_weights = weights[num_pairs:]
        rad_safe = np.where(rad > 1e-10, rad, 1.0)
        # d_bound = 1 - |pts|
        # grad(d_bound) = -pts / |pts|
        bound_grads = -(pts / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        return obj, grads

    def optimize_soft_min(pts, alpha_init, beta_init, beta_end, cooling, time_budget, 
                         stall_threshold=100, noise_scale_base=0.01, polish=False):
        """
        Main optimization loop using soft-min gradient ascent.
        """
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        alpha = alpha_init
        beta = beta_init
        stall_count = 0
        end_time = start_time + time_budget
        
        if polish:
            stall_threshold = 50
            cooling = 0.9995
        
        while time.time() < end_time:
            # Compute current min for offset
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            sq_dists = np.sum(diff**2, axis=2)
            np.fill_diagonal(sq_dists, np.inf)
            rad = np.linalg.norm(pts, axis=1)
            bound_dists = 1.0 - rad
            d_pair = np.sqrt(sq_dists[i_idx, j_idx]) / 2.0
            all_vals = np.concatenate([d_pair, bound_dists])
            current_min = np.min(all_vals)
            
            if current_min > best_score:
                best_score = current_min
                best_pts = pts.copy()
                stall_count = 0
            else:
                stall_count += 1
            
            # Compute gradient
            _, grads = soft_min_objective_and_grad(pts, beta, current_min)
            
            # Normalize gradient to control step size
            grad_norm = np.linalg.norm(grads)
            if grad_norm > 1e-10:
                grads /= grad_norm
            
            # Update
            step_size = alpha
            pts += step_size * grads
            
            # Project back inside circle
            rad_new = np.linalg.norm(pts, axis=1)
            mask_out = rad_new > 1.0
            if np.any(mask_out):
                pts[mask_out] /= rad_new[mask_out, np.newaxis]
            
            # Cool down
            alpha *= cooling
            
            # Increase beta (sharpen the minimum)
            progress = min(1.0, (time.time() - start_time) / time_budget)
            if polish:
                beta = beta_init + (beta_end - beta_init) * (progress ** 2.0)
            else:
                beta = beta_init + (beta_end - beta_init) * (progress ** 1.5)
            
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

    def polish_with_scipy(pts, time_budget, beta=500.0):
        """
        Use scipy minimize to polish the packing.
        We minimize the negative soft-min with a high k value.
        """
        end_time = start_time + time_budget
        current_best = compute_score(pts)
        
        def objective(x):
            pts_x = x.reshape(N, 2)
            # Use a fixed offset to keep numbers stable
            # We use the current best score as offset
            obj, _ = soft_min_objective_and_grad(pts_x, beta, current_best)
            return -obj # We want to maximize obj, so minimize -obj
        
        def gradient(x):
            pts_x = x.reshape(N, 2)
            _, grad = soft_min_objective_and_grad(pts_x, beta, current_best)
            return -grad.flatten() # Gradient of -obj
        
        x0 = pts.flatten()
        
        try:
            result = minimize(
                objective,
                x0,
                jac=gradient,
                method='L-BFGS-B',
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 50, 'ftol': 1e-16, 'gtol': 1e-16, 'maxls': 30}
            )
            pts_result = result.x.reshape(N, 2)
            
            # Project back inside
            rad = np.linalg.norm(pts_result, axis=1)
            mask_out = rad > 1.0
            if np.any(mask_out):
                pts_result[mask_out] /= rad[mask_out, np.newaxis]
            
            score_result = compute_score(pts_result)
            return pts_result, score_result
        except Exception:
            return pts, current_best

    # Generate multiple initializations
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Try hexagonal lattice with various spacings
    base_spacing = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    spacing_multipliers = [0.98, 1.0, 1.02]
    
    time_per_init = min(20.0, time_limit * 0.25)
    
    for idx, mult in enumerate(spacing_multipliers):
        if time.time() - start_time > time_limit * 0.60:
            break
        
        spacing = base_spacing * mult
        pts = hex_lattice_init(spacing)
        
        pts_opt, score = optimize_soft_min(pts, 
                                           alpha_init=0.015, 
                                           beta_init=10.0, 
                                           beta_end=100.0, 
                                           cooling=0.9998,
                                           time_budget=time_budget,
                                           stall_threshold=100,
                                           noise_scale_base=0.005)
        
        if score > best_score:
            best_score = score
            best_pts = pts_opt.copy()
    
    # Phase 2: Polish the best result
    if best_pts is not None and time.time() - start_time < time_limit * 0.80:
        remaining_time = time_limit * 0.95 - time.time()
        if remaining_time > 5.0:
            # First polish pass with soft-min gradient ascent
            pts_polished, score_polished = optimize_soft_min(best_pts,
                                                             alpha_init=0.005,
                                                             beta_init=50.0,
                                                             beta_end=200.0,
                                                             cooling=0.9995,
                                                             time_budget=min(remaining_time * 0.3, 10.0),
                                                             stall_threshold=50,
                                                             noise_scale_base=0.002,
                                                             polish=True)
            
            if score_polished > best_score:
                best_score = score_polished
                best_pts = pts_polished
            
            # Second: Use scipy for high-precision polish
            remaining_time2 = time_limit * 0.95 - time.time()
            if remaining_time2 > 3.0:
                pts_scipy, score_scipy = polish_with_scipy(best_pts, min(remaining_time2 * 0.5, 15.0), beta=500.0)
                
                if score_scipy > best_score:
                    best_score = score_scipy
                    best_pts = pts_scipy
            
            # Third: Another polish pass if time allows
            remaining_time3 = time_limit * 0.95 - time.time()
            if remaining_time3 > 2.0:
                pts_polished2, score_polished2 = optimize_soft_min(best_pts,
                                                                   alpha_init=0.002,
                                                                   beta_init=200.0,
                                                                   beta_end=500.0,
                                                                   cooling=0.999,
                                                                   time_budget=min(remaining_time3 * 0.5, 5.0),
                                                                   stall_threshold=20,
                                                                   noise_scale_base=0.001,
                                                                   polish=True)
                
                if score_polished2 > best_score:
                    best_score = score_polished2
                    best_pts = pts_polished2
    
    # Final verification
    if best_pts is None:
        best_pts = hex_lattice_init(base_spacing)
    
    # Ensure all points are inside the circle
    rad = np.linalg.norm(best_pts, axis=1)
    mask_out = rad > 1.0
    if np.any(mask_out):
        best_pts[mask_out] /= rad[mask_out, np.newaxis]
    
    return best_pts
