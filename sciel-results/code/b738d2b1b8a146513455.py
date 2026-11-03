"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s buffer for 150s limit
    
    # Initial guess: Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    def min_radius(c):
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        
        return min(min_pair_dist / 2.0, min_wall_dist)
    
    best_centers = centers.copy()
    best_r = min_radius(centers)
    
    # Soft-min objective using log-sum-exp
    def soft_min_radius(x, beta):
        c = x.reshape(N, 3)
        
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        
        # Wall distances
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        
        # Combine all constraints: min of (d_ij/2) and (wall_dist_i)
        # We want to maximize the minimum, so minimize -soft_min
        # soft_min of values v_i = -1/beta * log(sum(exp(-beta * v_i)))
        
        pair_half = d / 2.0
        # Stack wall distances as a vector
        all_vals = np.concatenate([pair_half[np.triu_indices(N, k=1)], dist_to_walls])
        
        # Soft min: -1/beta * log(sum(exp(-beta * v)))
        # For numerical stability, use log-sum-exp trick
        max_val = np.max(all_vals)
        exp_vals = np.exp(-beta * (all_vals - max_val))
        log_sum = np.log(np.sum(exp_vals))
        soft_min = max_val - log_sum / beta
        
        return -soft_min  # We minimize this
    
    def soft_min_radius_grad(x, beta):
        c = x.reshape(N, 3)
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        
        pair_half = d / 2.0
        all_vals = np.concatenate([pair_half[np.triu_indices(N, k=1)], dist_to_walls])
        
        max_val = np.max(all_vals)
        exp_vals = np.exp(-beta * (all_vals - max_val))
        weights = exp_vals / np.sum(exp_vals)
        
        # Gradient of soft_min w.r.t. all_vals is -weights
        # We need gradient w.r.t. x
        
        grad_x = np.zeros_like(x)
        
        # Pair contributions
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                w = weights[idx]
                if w > 1e-15:
                    vec = c[i] - c[j]
                    norm = d[i, j]
                    if norm > 1e-9:
                        # d/d(c_i) of (|c_i - c_j|/2) = (c_i - c_j) / |c_i - c_j|
                        # d/d(c_j) of (|c_i - c_j|/2) = -(c_i - c_j) / |c_i - c_j|
                        grad_x[i*3:(i+1)*3] += w * vec / norm
                        grad_x[j*3:(j+1)*3] -= w * vec / norm
                idx += 1
        
        # Wall contributions
        for i in range(N):
            w = weights[idx]
            if w > 1e-15:
                for k in range(3):
                    if c[i, k] > 0:
                        # Wall at 0.5, distance = 0.5 - c[i,k], grad = -1
                        grad_x[i*3+k] += w * (-1.0)
                    else:
                        # Wall at -0.5, distance = c[i,k] + 0.5, grad = +1
                        grad_x[i*3+k] += w * (1.0)
            idx += 1
        
        # We're minimizing -soft_min, so gradient is -grad(soft_min)
        # grad(soft_min) w.r.t. x = sum(weights * grad(vals))
        # So grad(-soft_min) = -sum(weights * grad(vals)) = -grad_x as computed above
        # Wait, let me re-derive:
        # soft_min = -1/beta * log(sum(exp(-beta * v_i)))
        # d(soft_min)/d(v_i) = exp(-beta*v_i) / sum(exp(-beta*v_j)) = weights[i]
        # d(soft_min)/dx = sum_i weights[i] * d(v_i)/dx
        # We minimize -soft_min, so gradient = -sum_i weights[i] * d(v_i)/dx = -grad_x
        return -grad_x
    
    # Continuation method: ramp up beta
    betas = [10, 50, 100, 200, 500, 1000, 2000]
    x = centers.flatten()
    
    for beta in betas:
        if time.time() - start_time > time_limit - 10:
            break
        
        # L-BFGS-B optimization
        try:
            res = minimize(
                fun=lambda x: soft_min_radius(x, beta),
                x0=x,
                jac=lambda x: soft_min_radius_grad(x, beta),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-9, 0.5 - 1e-9)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            x = res.x
        except Exception:
            pass
        
        # Update best
        c = x.reshape(N, 3)
        r = min_radius(c)
        if r > best_r:
            best_r = r
            best_centers = c.copy()
    
    # Final polish with higher beta
    if time.time() - start_time < time_limit - 5:
        beta_final = 5000
        try:
            res = minimize(
                fun=lambda x: soft_min_radius(x, beta_final),
                x0=x,
                jac=lambda x: soft_min_radius_grad(x, beta_final),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-9, 0.5 - 1e-9)] * (N * 3),
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
            )
            x = res.x
            c = x.reshape(N, 3)
            r = min_radius(c)
            if r > best_r:
                best_r = r
                best_centers = c.copy()
        except Exception:
            pass
    
    return best_centers
# EVOLVE-BLOCK-END
