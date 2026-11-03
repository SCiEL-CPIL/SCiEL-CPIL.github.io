"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Initial guess: 7x7 grid (49 points) + 1 point at (-2.5/7, -2.5/7)
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    extra_point = np.array([-2.5/7, -2.5/7])
    points = np.vstack([grid, extra_point])
    
    k = 200.0 # sharpness of softmin
    
    def objective_and_grad(x):
        if time.time() > deadline:
            return 1e9, np.zeros_like(x)
        pts = x.reshape(-1, 2)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Wall distances
        dw_x_pos = 0.5 - pts[:, 0]
        dw_x_neg = 0.5 + pts[:, 0]
        dw_y_pos = 0.5 - pts[:, 1]
        dw_y_neg = 0.5 + pts[:, 1]
        
        wall_dists = np.vstack([dw_x_pos, dw_x_neg, dw_y_pos, dw_y_neg])
        
        pair_dists = dists.flatten()
        all_dists = np.concatenate([pair_dists, wall_dists.flatten()])
        
        all_dists = np.clip(all_dists, 1e-6, 1.0)
        
        weights = np.exp(-k * all_dists)
        sum_weights = np.sum(weights)
        
        grad_pts = np.zeros_like(pts)
        
        weights_pair = weights[:N*N].reshape(N, N)
        
        mask = ~np.eye(N, dtype=bool)
        safe_dists = np.where(mask, dists, 1.0)
        
        factors = weights_pair / sum_weights / safe_dists
        np.fill_diagonal(factors, 0.0)
        
        sum_factors = np.sum(factors, axis=1)
        weighted_pts = np.sum(factors[:, :, np.newaxis] * pts[np.newaxis, :, :], axis=1)
        
        grad_pts += pts * sum_factors[:, np.newaxis] - weighted_pts
        
        weights_wall = weights[N*N:].reshape(4, N)
        norm_weights_wall = weights_wall / sum_weights
        
        grad_pts[:, 0] += -np.sum(norm_weights_wall[0]) + np.sum(norm_weights_wall[1])
        grad_pts[:, 1] += -np.sum(norm_weights_wall[2]) + np.sum(norm_weights_wall[3])
        
        softmin_val = -np.log(sum_weights) / k
        val = -softmin_val
        grad = -grad_pts.flatten()
        return val, grad

    bounds = [(-0.5, 0.5)] * (N * 2)
    
    try:
        res = minimize(objective_and_grad, points.flatten(), jac=True, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-5})
    except Exception:
        pass
    
    if hasattr(res, 'x') and res.x is not None:
        best_points = res.x.reshape(-1, 2)
    else:
        best_points = points
        
    best_points = np.clip(best_points, -0.4999, 0.4999)
    
    return best_points
# EVOLVE-BLOCK-END
