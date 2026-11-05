import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    start_time = time.time()
    timeout = 140.0 # 10s margin

    n = 8
    
    def log_product(points):
        # points: (n, 2)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Diameter
        max_dist = np.max(dists)
        if max_dist < 1e-10:
            return -np.inf
            
        # Scale to diameter 2
        scaled_dists = dists * (2.0 / max_dist)
        
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        off_diag = scaled_dists[mask]
        
        # Product of distances
        # Use log to avoid underflow/overflow
        return np.sum(np.log(off_diag))

    def objective(x):
        points = x.reshape(n, 2)
        lp = log_product(points)
        return -lp

    # 1. Baseline: Regular Octagon
    angles = 2 * np.pi * np.arange(n) / n
    reg_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score = log_product(reg_oct)
    best_points = reg_oct.copy()

    # 2. Local Optimization from diverse starts
    starts = [reg_oct.flatten()]
    
    # Two squares
    for rot in [0.1, 0.2, 0.3, 0.4, 0.5]:
        a1 = 2 * np.pi * np.arange(4) / 4
        a2 = 2 * np.pi * (np.arange(4) / 4 + rot)
        sq1 = np.column_stack([np.cos(a1), np.sin(a1)])
        sq2 = np.column_stack([np.cos(a2), np.sin(a2)])
        starts.append(np.vstack([sq1, sq2]).flatten())
        
    # Random
    np.random.seed(42)
    for _ in range(10):
        pts = np.random.uniform(-1, 1, (n, 2))
        starts.append(pts.flatten())
        
    for s in starts:
        if time.time() - start_time > timeout - 10:
            break
        try:
            res = minimize(objective, s, method='BFGS', options={'maxiter': 200, 'gtol': 1e-10})
            if res.fun < np.inf:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = res.x.reshape(n, 2)
        except:
            pass

    # 3. Global Optimization: Differential Evolution
    if time.time() - start_time < timeout - 20:
        bounds = [(-10, 10)] * (2 * n)
        try:
            de_res = differential_evolution(
                objective, 
                bounds, 
                seed=42, 
                maxiter=500, 
                popsize=20, 
                tol=1e-10, 
                polish=True
            )
            if de_res.fun < np.inf:
                score = -de_res.fun
                if score > best_score:
                    best_score = score
                    best_points = de_res.x.reshape(n, 2)
        except:
            pass

    # Final Refinement of the best found
    if time.time() - start_time < timeout - 5:
        try:
            final_res = minimize(objective, best_points.flatten(), method='BFGS', options={'maxiter': 1000, 'gtol': 1e-12})
            if final_res.fun < np.inf:
                score = -final_res.fun
                if score > best_score:
                    best_score = score
                    best_points = final_res.x.reshape(n, 2)
        except:
            pass

    # Validation
    dists = np.sqrt(np.sum((best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        best_points = reg_oct
        
    # Scale/Check bounds
    max_coord = np.max(np.abs(best_points))
    if max_coord > 1e6:
        best_points = best_points / max_coord * 1e5

    return [[float(p[0]), float(p[1])] for p in best_points]
