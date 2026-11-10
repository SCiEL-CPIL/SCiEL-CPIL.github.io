import numpy as np
import time
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    TIME_LIMIT = 600.0
    SAFETY_MARGIN = 10.0
    N = 16

    def get_cost_fast(pts):
        """Calculate (d_max / d_min)^2. Lower is better. Optimized for small N."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        
        mask = ~np.eye(pts.shape[0], dtype=bool)
        
        dist_sq_min = np.where(mask, dist_sq, np.inf)
        d_min_sq = np.min(dist_sq_min)
        d_max_sq = np.max(dist_sq)
        
        if d_min_sq <= 1e-12 or d_max_sq <= 1e-12:
            return np.inf
            
        return d_max_sq / d_min_sq

    def hexagonal_lattice_init(n_points=16):
        """Generate a hexagonal lattice and select the n_points closest to the centroid."""
        R = 10
        points = []
        a = 1.0
        b = np.sqrt(3) / 2
        for i in range(-R, R + 1):
            for j in range(-R, R + 1):
                x = i * a + j * 0.5 * a
                y = j * b
                points.append([x, y])
        
        points = np.array(points)
        
        centroid = np.mean(points, axis=0)
        dists = np.linalg.norm(points - centroid, axis=1)
        idx = np.argsort(dists)
        selected = points[idx[:n_points]]
        
        selected = selected - np.mean(selected, axis=0)
        return selected

    def grid_init(n_points=16):
        """Generate a square grid."""
        side = int(np.ceil(np.sqrt(n_points)))
        g = np.array([(i, j) for i in range(side) for j in range(side)], dtype=float)
        g = g - np.mean(g, axis=0)
        if len(g) > n_points:
            centroid = np.mean(g, axis=0)
            dists = np.linalg.norm(g - centroid, axis=1)
            idx = np.argsort(dists)
            g = g[idx[:n_points]]
        return g

    def random_init(n_points=16):
        return np.random.randn(n_points, 2)

    # Generate initial candidates
    candidates = [
        hexagonal_lattice_init(16),
        grid_init(16),
        random_init(16)
    ]
    
    best_pts = None
    best_cost = np.inf
    for c in candidates:
        c = c - np.mean(c, axis=0)
        cost = get_cost_fast(c)
        if cost < best_cost:
            best_cost = cost
            best_pts = c.copy()

    def simulated_annealing(initial_pts, time_budget):
        pts = initial_pts.copy()
        current_cost = get_cost_fast(pts)
        best_pts_sa = pts.copy()
        best_cost_sa = current_cost
        
        T = 0.3
        cooling_rate = 0.99995
        min_T = 1e-10
        no_improve_count = 0
        max_no_improve = 15000
        
        t_start = time.time()
        iter_count = 0
        
        while time.time() - t_start < time_budget:
            iter_count += 1
            
            idx = np.random.randint(0, len(pts))
            step_size = T * 0.5
            noise = np.random.randn(2) * step_size
            old_pos = pts[idx].copy()
            pts[idx] += noise
            
            new_cost = get_cost_fast(pts)
            
            if new_cost < current_cost:
                current_cost = new_cost
                no_improve_count = 0
                
                if current_cost < best_cost_sa:
                    best_cost_sa = current_cost
                    best_pts_sa = pts.copy()
            else:
                delta = new_cost - current_cost
                if T > min_T:
                    if delta > T * 100:
                        accept = False
                    else:
                        prob = np.exp(-delta / T)
                        accept = np.random.rand() < prob
                else:
                    accept = False
                
                if accept:
                    current_cost = new_cost
                else:
                    pts[idx] = old_pos
                    no_improve_count += 1
                    
            T *= cooling_rate
            
            if no_improve_count > max_no_improve:
                pts = best_pts_sa.copy()
                current_cost = best_cost_sa
                T = 0.05
                no_improve_count = 0
                max_no_improve = 8000

        return best_pts_sa, best_cost_sa

    time_elapsed = time.time() - start_time
    sa_time_budget = TIME_LIMIT - time_elapsed - SAFETY_MARGIN - 30.0
    if sa_time_budget > 15:
        best_pts_sa, best_cost_sa = simulated_annealing(best_pts, sa_time_budget)
        if best_cost_sa < best_cost:
            best_pts = best_pts_sa
            best_cost = best_cost_sa

        time_elapsed = time.time() - start_time
        remaining = TIME_LIMIT - time_elapsed - SAFETY_MARGIN - 30.0
        if remaining > 15:
            c2 = grid_init(16) + np.random.randn(16, 2) * 0.1
            c2 = c2 - np.mean(c2, axis=0)
            best_pts_sa2, best_cost_sa2 = simulated_annealing(c2, remaining)
            if best_cost_sa2 < best_cost:
                best_pts = best_pts_sa2
                best_cost = best_cost_sa2

    time_elapsed = time.time() - start_time
    polish_budget = TIME_LIMIT - time_elapsed - SAFETY_MARGIN
    
    if polish_budget > 8:
        def objective(params):
            pts = params.reshape(16, 2)
            return get_cost_fast(pts)

        initial_params = best_pts.flatten()
        
        try:
            res = minimize(objective, initial_params, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            if hasattr(res, 'fun') and np.isfinite(res.fun):
                res_pts = res.x.reshape(16, 2)
                res_cost = get_cost_fast(res_pts)
                if res_cost < best_cost:
                    best_pts = res_pts
                    best_cost = res_cost
        except Exception:
            pass

    # Final high-precision polish pass using a smooth surrogate (log-sum-exp)
    time_elapsed = time.time() - start_time
    final_polish_budget = TIME_LIMIT - time_elapsed - SAFETY_MARGIN
    
    if final_polish_budget > 10:
        def smooth_objective_and_grad(params):
            pts = params.reshape(16, 2)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=-1)
            mask = ~np.eye(16, dtype=bool)
            dist_sq_masked = np.where(mask, dist_sq, 0.0)
            
            valid_dsq = dist_sq_masked[mask]
            if len(valid_dsq) == 0:
                return 1e10, np.zeros_like(params)
            
            median_dsq = np.median(valid_dsq)
            beta = 0.1 * median_dsq
            beta = max(beta, 1e-3)
            
            # Numerically stable log-sum-exp
            scaled_max = dist_sq_masked / beta
            max_scaled = np.max(scaled_max)
            log_sum_exp_max = beta * (max_scaled + np.log(np.sum(np.exp(scaled_max - max_scaled))))
            
            scaled_min = -dist_sq_masked / beta
            min_scaled = np.max(scaled_min)
            log_sum_exp_min = -beta * (min_scaled + np.log(np.sum(np.exp(scaled_min - min_scaled))))
            
            d_min_sq_smooth = log_sum_exp_min
            d_max_sq_smooth = log_sum_exp_max
            
            if d_min_sq_smooth <= 1e-12 or d_max_sq_smooth <= 1e-12:
                return 1e10, np.zeros_like(params)
            
            cost = d_max_sq_smooth / d_min_sq_smooth
            
            # Gradients with numerically stable computation
            exp_max = np.exp(scaled_max - max_scaled)
            sum_exp_max = np.sum(exp_max)
            d_max_grad_dsq = exp_max / sum_exp_max
            
            exp_min = np.exp(scaled_min - min_scaled)
            sum_exp_min = np.sum(exp_min)
            d_min_grad_dsq = -exp_min / sum_exp_min
            
            d_cost_d_dsq = d_max_grad_dsq / d_min_sq_smooth - (d_max_sq_smooth / (d_min_sq_smooth ** 2)) * d_min_grad_dsq
            
            grad = np.zeros((16, 2))
            for i in range(16):
                for j in range(16):
                    if i != j:
                        w = d_cost_d_dsq[i, j]
                        if abs(w) > 1e-15:
                            diff_ij = pts[i] - pts[j]
                            grad[i] += 2.0 * w * diff_ij
                            grad[j] -= 2.0 * w * diff_ij
            
            return cost, grad

        def smooth_objective(params):
            cost, _ = smooth_objective_and_grad(params)
            return cost

        def smooth_grad(params):
            _, grad = smooth_objective_and_grad(params)
            return grad

        initial_params = best_pts.flatten()
        
        try:
            res = minimize(smooth_objective, initial_params, jac=smooth_grad, method='L-BFGS-B', 
                           options={'maxiter': 3000, 'gtol': 1e-12, 'ftol': 1e-14, 'maxls': 50})
            
            if hasattr(res, 'fun') and np.isfinite(res.fun):
                res_pts = res.x.reshape(16, 2)
                res_cost = get_cost_fast(res_pts)
                if res_cost < best_cost:
                    best_pts = res_pts
                    best_cost = res_cost
        except Exception:
            pass
        
        # Second polish pass with tighter tolerance
        time_elapsed2 = time.time() - start_time
        second_polish_budget = TIME_LIMIT - time_elapsed2 - SAFETY_MARGIN
        
        if second_polish_budget > 5:
            try:
                res2 = minimize(smooth_objective, best_pts.flatten(), jac=smooth_grad, method='L-BFGS-B', 
                               options={'maxiter': 5000, 'gtol': 1e-14, 'ftol': 1e-16, 'maxls': 30})
                
                if hasattr(res2, 'fun') and np.isfinite(res2.fun):
                    res_pts2 = res2.x.reshape(16, 2)
                    res_cost2 = get_cost_fast(res_pts2)
                    if res_cost2 < best_cost:
                        best_pts = res_pts2
                        best_cost = res_cost2
            except Exception:
                pass

    final_pts = best_pts
    if final_pts is None:
        final_pts = np.random.randn(16, 2)
        
    if not np.all(np.isfinite(final_pts)):
        final_pts = grid_init(16)

    return final_pts
