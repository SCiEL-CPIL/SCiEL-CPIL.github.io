import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    TIME_LIMIT = 600.0
    SAFETY_MARGIN = 30.0
    N = 16
    N_PAIRS = N * (N - 1) // 2
    i_idx, j_idx = np.triu_indices(N, k=1)

    def get_ratio(pts):
        """Calculate (d_min / d_max)^2. Higher is better."""
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max <= 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def compute_dist_matrix(pts):
        """Compute full pairwise distance matrix."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        dist = np.sqrt(np.maximum(dist_sq, 0.0))
        return dist

    def objective_and_grad(x, beta):
        """
        Objective: log(d_max) - log(d_min) using soft min/max.
        Gradient computed analytically.
        """
        pts = x.reshape(N, 2)
        dist = compute_dist_matrix(pts)
        d_pairs = dist[i_idx, j_idx]
        
        if np.any(d_pairs < 1e-15):
            return 1e10, np.zeros_like(x)

        d_max_val = np.max(d_pairs)
        d_min_val = np.min(d_pairs)
        
        # Soft max
        shifted_max = beta * (d_pairs - d_max_val)
        w_max = np.exp(shifted_max)
        w_max /= np.sum(w_max)
        soft_max_val = d_max_val + np.log(np.sum(np.exp(shifted_max))) / beta
        
        # Soft min
        shifted_min = -beta * (d_pairs - d_min_val)
        w_min = np.exp(shifted_min)
        w_min /= np.sum(w_min)
        soft_min_val = d_min_val - np.log(np.sum(np.exp(shifted_min))) / beta
        
        if soft_min_val < 1e-15 or soft_max_val < 1e-15:
            return 1e10, np.zeros_like(x)
            
        obj_val = np.log(soft_max_val) - np.log(soft_min_val)
        
        grad = np.zeros_like(x)
        
        for k in range(N_PAIRS):
            i = i_idx[k]
            j = j_idx[k]
            d_ij = dist[i, j]
            if d_ij < 1e-15:
                continue
            diff_ij = pts[i] - pts[j]
            inv_d = 1.0 / d_ij
            
            # Contribution to log(soft_max)
            grad[2*i] += w_max[k] * (diff_ij[0] * inv_d) / soft_max_val
            grad[2*i+1] += w_max[k] * (diff_ij[1] * inv_d) / soft_max_val
            grad[2*j] += w_max[k] * (-diff_ij[0] * inv_d) / soft_max_val
            grad[2*j+1] += w_max[k] * (-diff_ij[1] * inv_d) / soft_max_val
            
            # Contribution to -log(soft_min)
            grad[2*i] -= w_min[k] * (diff_ij[0] * inv_d) / soft_min_val
            grad[2*i+1] -= w_min[k] * (diff_ij[1] * inv_d) / soft_min_val
            grad[2*j] -= w_min[k] * (-diff_ij[0] * inv_d) / soft_min_val
            grad[2*j+1] -= w_min[k] * (-diff_ij[1] * inv_d) / soft_min_val
        
        return obj_val, grad

    def ring_init(n_inner, n_outer, r_inner, r_outer):
        pts = []
        if n_inner > 0:
            for i in range(n_inner):
                angle = 2 * np.pi * i / n_inner
                pts.append([r_inner * np.cos(angle), r_inner * np.sin(angle)])
        if n_outer > 0:
            for i in range(n_outer):
                angle = 2 * np.pi * i / n_outer + np.pi / n_outer
                pts.append([r_outer * np.cos(angle), r_outer * np.sin(angle)])
        return np.array(pts)

    def grid_init():
        pts = []
        for i in range(4):
            for j in range(4):
                pts.append([i, j])
        pts = np.array(pts, dtype=float)
        pts = pts - np.mean(pts, axis=0)
        return pts

    def hex_init():
        pts = []
        a = 1.0
        b = np.sqrt(3) / 2
        pts.append([0, 0])
        for i in range(6):
            angle = 2 * np.pi * i / 6
            pts.append([a * np.cos(angle), a * np.sin(angle)])
        ring2_pts = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * a + j * 0.5 * a
                y = j * b
                r = np.sqrt(x*x + y*y)
                if 1.5 * a - 0.1 < r < 2.5 * a + 0.1:
                    ring2_pts.append([x, y])
        ring2_pts = np.array(ring2_pts)
        if len(ring2_pts) >= 9:
            r_target = 2.0 * a
            dists_to_target = np.abs(np.linalg.norm(ring2_pts, axis=1) - r_target)
            idx = np.argsort(dists_to_target)[:9]
            ring2_pts = ring2_pts[idx]
        pts = np.array(pts)
        pts = np.vstack([pts, ring2_pts])
        pts = pts - np.mean(pts, axis=0)
        return pts

    def circle_init(n=16):
        pts = []
        for i in range(n):
            angle = 2 * np.pi * i / n
            pts.append([np.cos(angle), np.sin(angle)])
        return np.array(pts)

    def generate_candidates():
        candidates = []
        candidates.append(ring_init(1, 15, 0, 1.0))
        candidates.append(ring_init(4, 12, 0.5, 1.0))
        candidates.append(ring_init(7, 9, 0.4, 1.0))
        candidates.append(ring_init(9, 7, 0.4, 1.0))
        candidates.append(ring_init(1, 3, 0, 0.5))
        for i, c in enumerate(candidates):
            c = c - np.mean(c, axis=0)
            max_r = np.max(np.linalg.norm(c, axis=1))
            if max_r > 0:
                c = c / max_r
            candidates[i] = c
        candidates.append(grid_init())
        candidates.append(hex_init())
        candidates.append(circle_init(16))
        return candidates

    best_pts = None
    best_ratio = 0.0

    for c in generate_candidates():
        if len(c) != 16:
            continue
        ratio = get_ratio(c)
        if ratio > best_ratio:
            best_ratio = ratio
            best_pts = c.copy()

    def optimize_from(pts, time_budget, beta_start=5.0, beta_end=50.0):
        """
        Optimize starting from pts, increasing beta for higher precision.
        """
        pts = pts.copy()
        t_start = time.time()
        x = pts.flatten()
        
        # Normalize
        x = x - np.mean(x.reshape(N, 2), axis=0).flatten()
        max_r = np.max(np.linalg.norm(x.reshape(N, 2), axis=1))
        if max_r > 0:
            x = x / max_r
            
        best_x = x.copy()
        best_obj = np.inf
        best_ratio_local = get_ratio(x.reshape(N, 2))
        
        betas = [beta_start, 10.0, 20.0, beta_end]
        
        for beta in betas:
            if time.time() - t_start > time_budget:
                break
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
                
            # Multiple restarts with small perturbations
            n_restarts = 2
            for r in range(n_restarts):
                if time.time() - t_start > time_budget:
                    break
                if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                    break
                    
                if r == 0:
                    x_start = best_x.copy()
                else:
                    x_start = best_x + np.random.randn(N * 2) * 0.01
                    
                try:
                    result = minimize(
                        lambda x: objective_and_grad(x, beta)[0],
                        x_start,
                        jac=lambda x: objective_and_grad(x, beta)[1],
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'maxls': 50, 'ftol': 1e-15, 'gtol': 1e-12}
                    )
                    obj_val, _ = objective_and_grad(result.x, beta)
                    if obj_val < best_obj:
                        best_obj = obj_val
                        best_x = result.x
                        best_ratio_local = get_ratio(result.x.reshape(N, 2))
                except Exception:
                    pass
                    
        return best_x.reshape(N, 2), best_ratio_local

    # Main optimization phase
    time_elapsed = time.time() - start_time
    main_budget = min(400.0, TIME_LIMIT - time_elapsed - SAFETY_MARGIN)
    if main_budget > 60:
        opt_pts, opt_ratio = optimize_from(best_pts, time_budget=main_budget)
        if opt_ratio > best_ratio:
            best_pts = opt_pts
            best_ratio = opt_ratio

    # Final polish phase with very high beta
    time_elapsed = time.time() - start_time
    polish_budget = min(100.0, TIME_LIMIT - time_elapsed - SAFETY_MARGIN)
    if polish_budget > 30:
        opt_pts, opt_ratio = optimize_from(best_pts, time_budget=polish_budget, beta_start=50.0, beta_end=100.0)
        if opt_ratio > best_ratio:
            best_pts = opt_pts
            best_ratio = opt_ratio

    # Ensure finite and correct shape
    final_pts = best_pts
    if final_pts is None or not np.all(np.isfinite(final_pts)) or final_pts.shape != (16, 2):
        final_pts = grid_init()

    return final_pts
