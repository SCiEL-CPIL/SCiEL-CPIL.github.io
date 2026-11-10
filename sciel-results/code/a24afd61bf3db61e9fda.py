import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

def _score(params):
    # params: (8,)
    # Returns negative log score for minimization
    pts_4 = params.reshape(4, 2)
    pts_8 = np.vstack([pts_4, -pts_4])
    
    # Pairwise distances
    # (8, 8, 2)
    diffs = pts_8[:, np.newaxis, :] - pts_8[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    
    diam = np.max(dists)
    if diam < 1e-12:
        return 1e10 # Penalize degenerate
        
    scaled_dists = dists / diam * 2.0
    
    # Mask diagonal
    mask = ~np.eye(8, dtype=bool)
    # log product
    log_prod = np.sum(np.log(scaled_dists[mask] + 1e-300))
    
    # Score = Delta / 8^8
    # log Score = log Prod - 8 log 8
    log_score = log_prod - 8 * np.log(8)
    
    return -log_score

def construct():
    deadline = time.time() + 140
    
    # Initial guess: Regular Octagon
    # 4 unique points
    R = 1.0
    v1 = [R, 0]
    v2 = [0, R]
    v3 = [R/np.sqrt(2), R/np.sqrt(2)]
    v4 = [-R/np.sqrt(2), R/np.sqrt(2)]
    x0 = np.array(v1 + v2 + v3 + v4)
    
    bounds = [(-5, 5)] * 8
    
    best_score = _score(x0)
    best_params = x0
    
    # Phase 1: Differential Evolution
    if time.time() < deadline - 20:
        try:
            result = differential_evolution(
                _score, bounds,
                maxiter=100, popsize=20, tol=1e-8,
                seed=42, polish=True,
                callback=lambda x, convergence: convergence or (time.time() > deadline - 20)
            )
            if result.success:
                s = result.fun
                if s < best_score:
                    best_score = s
                    best_params = result.x
        except Exception:
            pass

    # Phase 2: Local Minimization
    if time.time() < deadline - 5:
        try:
            res = minimize(
                _score, best_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.success:
                s = res.fun
                if s < best_score:
                    best_score = s
                    best_params = res.x
        except Exception:
            pass

    # Convert back to 8 points
    pts_4 = best_params.reshape(4, 2)
    pts_8 = np.vstack([pts_4, -pts_4])
    
    # Ensure coordinates are within bounds
    pts_8 = np.clip(pts_8, -1e6, 1e6)
    
    return pts_8.tolist()
