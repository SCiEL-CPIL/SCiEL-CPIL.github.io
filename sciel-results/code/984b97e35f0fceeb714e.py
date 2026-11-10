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
        
        T = 0.5
        cooling_rate = 0.9999
        min_T = 1e-8
        max_iter = 10000000
        no_improve_count = 0
        max_no_improve = 20000
        
        t_start = time.time()
        
        for i in range(max_iter):
            if time.time() - t_start > time_budget:
                break
            
            idx = np.random.randint(0, len(pts))
            noise = np.random.randn(2) * T
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
                if delta < 0:
                    accept = True
                else:
                    if T > min_T:
                        if delta > T * 50: 
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
                T = 0.1
                no_improve_count = 0
                max_no_improve = 10000

        return best_pts_sa, best_cost_sa

    time_elapsed = time.time() - start_time
    sa_time_budget = TIME_LIMIT - time_elapsed - SAFETY_MARGIN - 30.0
    if sa_time_budget > 10:
        best_pts_sa, best_cost_sa = simulated_annealing(best_pts, sa_time_budget)
        if best_cost_sa < best_cost:
            best_pts = best_pts_sa
            best_cost = best_cost_sa

        time_elapsed = time.time() - start_time
        remaining = TIME_LIMIT - time_elapsed - SAFETY_MARGIN - 30.0
        if remaining > 30:
            c2 = grid_init(16) + np.random.randn(16, 2) * 0.1
            c2 = c2 - np.mean(c2, axis=0)
            best_pts_sa2, best_cost_sa2 = simulated_annealing(c2, remaining)
            if best_cost_sa2 < best_cost:
                best_pts = best_pts_sa2
                best_cost = best_cost_sa2

    time_elapsed = time.time() - start_time
    polish_budget = TIME_LIMIT - time_elapsed - SAFETY_MARGIN
    
    if polish_budget > 5:
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
    
    if final_polish_budget > 5:
        def smooth_objective(params):
            pts = params.reshape(16, 2)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=-1)
            mask = ~np.eye(pts.shape[0], dtype=bool)
            dist_sq = np.where(mask, dist_sq, 0)
            
            # Soft max and soft min using log-sum-exp trick for stability
            # We want to minimize (d_max / d_min)^2
            # Let's work with log distances or just squares directly with scaling
            # To avoid overflow in exp, subtract max
            max_sq = np.max(dist_sq)
            min_sq = np.min(dist_sq[dist_sq > 1e-12]) if np.any(dist_sq > 1e-12) else 1e-12
            
            # Scale for stability in log-sum-exp
            # We want to approximate min(d_sq) and max(d_sq)
            # log(sum(exp(x/beta))) * beta approximates max(x)
            # log(sum(exp(-x/beta))) * -beta approximates min(x)
            
            beta = 0.1 # Sharpness parameter
            
            # Soft Max
            exp_sq = np.exp(dist_sq / beta)
            log_sum_exp_max = beta * np.log(np.sum(exp_sq))
            
            # Soft Min
            exp_neg_sq = np.exp(-dist_sq / beta)
            log_sum_exp_min = -beta * np.log(np.sum(exp_neg_sq))
            
            # Approximate d_min_sq and d_max_sq
            d_min_sq_smooth = log_sum_exp_min
            d_max_sq_smooth = log_sum_exp_max
            
            if d_min_sq_smooth <= 1e-12 or d_max_sq_smooth <= 1e-12:
                return 1e10
                
            return d_max_sq_smooth / d_min_sq_smooth

        initial_params = best_pts.flatten()
        
        try:
            # Use BFGS or L-BFGS-B for smooth objective
            res = minimize(smooth_objective, initial_params, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'gtol': 1e-8, 'ftol': 1e-10})
            
            if hasattr(res, 'fun') and np.isfinite(res.fun):
                res_pts = res.x.reshape(16, 2)
                res_cost = get_cost_fast(res_pts)
                if res_cost < best_cost:
                    best_pts = res_pts
                    best_cost = res_cost
        except Exception:
            pass

    final_pts = best_pts
    if final_pts is None:
        final_pts = np.random.randn(16, 2)
        
    if not np.all(np.isfinite(final_pts)):
        final_pts = grid_init(16)

    return final_pts
