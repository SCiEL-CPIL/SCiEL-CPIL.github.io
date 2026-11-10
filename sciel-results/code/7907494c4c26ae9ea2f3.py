import numpy as np
import time
from scipy.optimize import minimize

N = 100


def _min_radius(centers):
    """Compute the maximum common radius for given centers inside [-0.5, 0.5]^3."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check inside cube
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # Distance to walls
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    r = min(min_pair / 2.0, min_wall)
    return max(r, 0.0)


def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin

    # Initial placement: a jittered grid
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].astype(np.float64)
    
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.001, size=centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    
    best_centers = centers.copy()
    best_r = _min_radius(best_centers)
    
    # Log-sum-exp optimization
    def obj_and_grad(x, beta):
        c = x.reshape(N, 3)
        
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1e-10)
        
        # Wall distances
        abs_c = np.abs(c)
        max_abs = np.max(abs_c, axis=1)
        w = 0.5 - max_abs
        
        # Calculate S
        exp_d = np.exp(-beta * dists)
        np.fill_diagonal(exp_d, 0.0)
        exp_w = np.exp(-beta * w)
        
        S = np.sum(exp_d) + np.sum(exp_w)
        
        val = (1.0 / beta) * np.log(S)
        
        # Gradient
        inv_dists = 1.0 / dists
        np.fill_diagonal(inv_dists, 0.0)
        
        pair_grad_factor = (exp_d * inv_dists)[:, :, np.newaxis]
        pair_grads = np.sum(diff * pair_grad_factor, axis=1)
        
        # Wall gradient
        is_max = (abs_c == max_abs[:, np.newaxis])
        wall_grad = -np.sign(c) * is_max
        
        grad_term = pair_grads + exp_w[:, np.newaxis] * wall_grad
        grad_val = - (1.0 / S) * grad_term
        
        return val, grad_val.reshape(N * 3)

    betas = [10, 20, 50, 100, 200, 500]
    x = centers.flatten()
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(obj_and_grad, x, args=(beta,), method='L-BFGS-B', 
                           jac=True, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success:
                x = res.x
                c = x.reshape(N, 3)
                c = np.clip(c, -0.4999, 0.4999)
                r = _min_radius(c)
                if r > best_r:
                    best_r = r
                    best_centers = c.copy()
        except Exception:
            pass
            
    # Final polish
    if time.time() < deadline - 5:
        try:
            res = minimize(obj_and_grad, x, args=(1000,), method='L-BFGS-B', 
                           jac=True, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.success:
                x = res.x
                c = x.reshape(N, 3)
                c = np.clip(c, -0.4999, 0.4999)
                r = _min_radius(c)
                if r > best_r:
                    best_r = r
                    best_centers = c.copy()
        except Exception:
            pass

    return best_centers
