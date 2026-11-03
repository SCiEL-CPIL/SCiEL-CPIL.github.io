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
    
    def soft_min_objective(x, k):
        """
        Smooth surrogate: -soft_min of all constraint values.
        We minimize this to maximize the minimum constraint.
        Uses log-sum-exp for numerical stability.
        """
        pts = x.reshape(N, 2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        
        pair_vals = np.sqrt(sq_dists[i_idx, j_idx]) / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        # Soft min: -1/k * log(sum(exp(-k * vals)))
        # For stability, shift by min value
        min_val = np.min(all_vals)
        shifted = all_vals - min_val
        exp_vals = np.exp(-k * shifted)
        log_sum = np.log(np.sum(exp_vals))
        soft_min = min_val - log_sum / k
        
        return -soft_min
    
    def soft_min_gradient(x, k):
        """
        Gradient of -soft_min objective.
        d(-soft_min)/dx = -d(soft_min)/dx
        soft_min = min_val - (1/k) * log(sum(exp(-k * (all_vals - min_val))))
        d(soft_min)/dx_i = -(1/k) * sum_j [exp(-k*(v_j - min_val)) / sum(exp(-k*(v_j - min_val)))] * dv_j/dx_i
                         = -sum_j [w_j * dv_j/dx_i] where w_j = exp(-k*(v_j - min_val)) / sum(exp(-k*(v_j - min_val)))
        So d(-soft_min)/dx_i = sum_j [w_j * dv_j/dx_i]
        """
        pts = x.reshape(N, 2)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        
        pair_vals = np.sqrt(sq_dists[i_idx, j_idx]) / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        # Compute weights
        min_val = np.min(all_vals)
        shifted = all_vals - min_val
        # Mask to avoid overflow
        mask_close = shifted < 30.0
        if not np.any(mask_close):
            weights = np.ones_like(all_vals) / len(all_vals)
        else:
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
            else:
                weights = np.ones_like(all_vals) / len(all_vals)
        
        # Compute gradients
        grads = np.zeros_like(pts)
        
        # Pair contributions: v_ij = |p_i - p_j| / 2
        # d(v_ij)/d(p_i) = (p_i - p_j) / (2 * |p_i - p_j|)
        # d(v_ij)/d(p_j) = (p_j - p_i) / (2 * |p_i - p_j|)
        pair_weights = weights[:num_pairs]
        d_pair_sq = sq_dists[i_idx, j_idx]
        d_pair = np.sqrt(d_pair_sq)
        mask_nonzero = d_pair > 1e-10
        if np.any(mask_nonzero):
            idx_m = mask_nonzero
            vecs = (diff[i_idx[idx_m], j_idx[idx_m]] / 
                   (2.0 * d_pair[idx_m, np.newaxis])) * pair_weights[idx_m, np.newaxis]
            
            np.add.at(grads, i_idx, vecs)
            np.add.at(grads, j_idx, -vecs)
        
        # Boundary contributions: v_i = 1 - |p_i|
        # d(v_i)/d(p_i) = -p_i / |p_i|
        bound_weights = weights[num_pairs:]
        rad_safe = np.where(rad > 1e-10, rad, 1.0)
        bound_grads = -(pts / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        return grads.flatten()
    
    def _hex_init(spacing):
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
    
    def _sa_optimize(pts, time_budget, k_start=5.0, k_end=100.0, alpha_start=0.01, alpha_end=0.0005):
        """
        Simulated Annealing optimization using soft-min gradient.
        """
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        n_iter = 0
        end_time = start_time + time_budget
        total_budget = time_budget
        
        while time.time() < end_time:
            n_iter += 1
            progress = min(1.0, n_iter / 5000.0)  # Normalize over expected iterations
            
            # Annealing schedule
            k = k_start + (k_end - k_start) * progress
            alpha = alpha_start * (alpha_end / alpha_start) ** progress
            
            # Compute objective and gradient
            obj = soft_min_objective(pts.flatten(), k)
            grad = soft_min_gradient(pts.flatten(), k)
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1.0:
                grad /= grad_norm
            
            # Update with some noise for exploration
            noise = np.random.normal(0, alpha * 0.1, size=pts.shape)
            pts += alpha * grad.reshape(N, 2) + noise
            
            # Project back inside circle
            rad_new = np.linalg.norm(pts, axis=1)
            mask_out = rad_new > 1.0
            if np.any(mask_out):
                pts[mask_out] /= rad_new[mask_out, np.newaxis]
            
            # Check score
            current_score = compute_score(pts)
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
        
        return best_pts, best_score
    
    def _polish(pts, time_budget):
        """
        Final polish using L-BFGS-B on the smooth soft-min objective.
        """
        if time.time() - start_time > time_limit * 0.95:
            return pts, compute_score(pts)
        
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        # Try multiple k values for polishing
        for k in [50.0, 100.0, 200.0]:
            if time.time() - start_time > time_limit * 0.95:
                break
            
            x0 = best_pts.flatten()
            
            def obj_func(x):
                if time.time() - start_time > time_limit * 0.95:
                    return np.inf
                return soft_min_objective(x, k)
            
            def grad_func(x):
                if time.time() - start_time > time_limit * 0.95:
                    return np.zeros_like(x)
                return soft_min_gradient(x, k)
            
            try:
                result = minimize(
                    obj_func, x0, jac=grad_func,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000}
                )
                
                if result.success or result.fun < obj_func(x0):
                    new_pts = result.x.reshape(N, 2)
                    # Project inside circle
                    rad_new = np.linalg.norm(new_pts, axis=1)
                    mask_out = rad_new > 1.0
                    if np.any(mask_out):
                        new_pts[mask_out] /= rad_new[mask_out, np.newaxis]
                    
                    new_score = compute_score(new_pts)
                    if new_score > best_score:
                        best_score = new_score
                        best_pts = new_pts.copy()
            except Exception:
                pass
        
        return best_pts, best_score
    
    # Main optimization pipeline
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Hexagonal lattice initialization with SA optimization
    base_spacing = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    spacing_multipliers = [0.97, 0.99, 1.0, 1.01, 1.03]
    
    time_per_init = min(18.0, time_limit * 0.16)
    
    for idx, mult in enumerate(spacing_multipliers):
        if time.time() - start_time > time_limit * 0.60:
            break
        
        spacing = base_spacing * mult
        pts = _hex_init(spacing)
        
        time_budget = time_per_init
        pts_opt, score = _sa_optimize(pts, time_budget, k_start=5.0, k_end=100.0)
        
        if score > best_score:
            best_score = score
            best_pts = pts_opt.copy()
    
    # Phase 2: Random initialization with SA
    if time.time() - start_time < time_limit * 0.75:
        for _ in range(2):
            if time.time() - start_time > time_limit * 0.75:
                break
            
            pts = []
            while len(pts) < N:
                p = np.random.uniform(-0.98, 0.98, 2)
                if np.linalg.norm(p) < 0.98:
                    pts.append(p)
            pts = np.array(pts)
            
            time_budget = min(15.0, time_limit * 0.12)
            pts_opt, score = _sa_optimize(pts, time_budget, k_start=4.0, k_end=80.0)
            
            if score > best_score:
                best_score = score
                best_pts = pts_opt.copy()
    
    # Phase 3: Polish the best result
    if best_pts is not None and time.time() - start_time < time_limit * 0.90:
        remaining_time = time_limit * 0.97 - time.time()
        if remaining_time > 3.0:
            pts_polished, score_polished = _polish(best_pts, remaining_time)
            
            if score_polished > best_score:
                best_score = score_polished
                best_pts = pts_polished
    
    # Fallback
    if best_pts is None:
        best_pts = _hex_init(base_spacing)
    
    # Ensure all points are inside the circle
    rad = np.linalg.norm(best_pts, axis=1)
    mask_out = rad > 1.0
    if np.any(mask_out):
        best_pts[mask_out] /= rad[mask_out, np.newaxis]
    
    return best_pts
