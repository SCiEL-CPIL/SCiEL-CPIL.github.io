import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = 8
    
    def compute_score(points):
        # points is shape (n, 2)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Avoid log(0)
        dists = np.maximum(dists, 1e-10)
        
        mask = ~np.eye(n, dtype=bool)
        max_dist = np.max(dists[mask])
        
        if max_dist == 0:
            return -np.inf
            
        log_prod_ordered = np.sum(np.log(dists[mask]))
        
        return log_prod_ordered - 56 * np.log(max_dist)

    def d4_points(params):
        # params: [x, y] with x > y > 0
        # D4 orbit of (x, y) has 8 points:
        # (x, y), (-x, y), (-x, -y), (x, -y)
        # (y, x), (-y, x), (-y, -x), (y, -x)
        x, y = params
        p1 = np.array([x, y])
        p2 = np.array([-x, y])
        p3 = np.array([-x, -y])
        p4 = np.array([x, -y])
        p5 = np.array([y, x])
        p6 = np.array([-y, x])
        p7 = np.array([-y, -x])
        p8 = np.array([y, -x])
        return np.vstack([p1, p2, p3, p4, p5, p6, p7, p8])

    def objective_2d(params):
        points = d4_points(params)
        s = compute_score(points)
        if not np.isfinite(s):
            return 1e10
        return -s

    best_score = -np.inf
    best_params = None
    best_points = None
    
    # Strategy: Grid search in D4 parameter space (x, y)
    # The regular octagon corresponds to x = cos(pi/8), y = sin(pi/8)
    # x ~ 0.9239, y ~ 0.3827
    # Let's search in a box around this.
    
    # Grid search
    # x in [0.5, 1.5], y in [0.1, x-0.01]
    # Use a coarse grid first, then refine around the best.
    
    # Coarse grid
    x_vals = np.linspace(0.5, 1.5, 100)
    
    for x in x_vals:
        if time.time() > deadline:
            break
        # y must be less than x to avoid duplicates in orbit, and > 0
        # Also, to avoid points being too close, y should be > 0.1
        y_max = x - 0.01
        if y_max <= 0.1:
            continue
        y_vals = np.linspace(0.1, y_max, 50)
        
        for y in y_vals:
            if time.time() > deadline:
                break
            params = np.array([x, y])
            score = compute_score(d4_points(params))
            if score > best_score:
                best_score = score
                best_params = params
                best_points = d4_points(params)
                
    # Local refinement using Nelder-Mead on the 2D parameter space
    if best_params is not None:
        try:
            from scipy.optimize import minimize
            res = minimize(objective_2d, best_params, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.success or res.fun < objective_2d(best_params):
                final_params = res.x
                # Ensure x > y > 0
                x, y = final_params
                if x > y > 0:
                    final_points = d4_points(final_params)
                    final_score = compute_score(final_points)
                    if final_score > best_score:
                        best_score = final_score
                        best_points = final_points
        except Exception:
            pass

    # Fallback to regular octagon if something went wrong
    if best_points is None:
        angles = 2 * np.pi * np.arange(n) / n
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return [[float(x), float(y)] for x, y in best_points]
