import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial configuration: 4x4x4 grid, take first 50
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c0 = grid[:N].copy()
    
    # Add small random perturbation
    np.random.seed(42)
    c0 += np.random.normal(0, 0.01, c0.shape)
    c0 = np.clip(c0, -0.49, 0.49)
    
    def energy(c):
        c = c.reshape(N, 3)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist_sq = np.where(dist_sq < 1e-12, 1e-12, dist_sq)
        mask = ~np.eye(N, dtype=bool)
        inv_dist4 = np.zeros_like(dist_sq)
        np.divide(1.0, dist_sq**2, out=inv_dist4, where=mask)
        return np.sum(inv_dist4[mask])
        
    def energy_grad(c):
        c = c.reshape(N, 3)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist_sq = np.where(dist_sq < 1e-12, 1e-12, dist_sq)
        mask = ~np.eye(N, dtype=bool)
        
        inv_dist6 = np.zeros_like(dist_sq)
        np.divide(1.0, dist_sq**3, out=inv_dist6, where=mask)
        
        # d/dx (1/d^4) = -4/d^6 * (x_i - x_j)
        grad = -4.0 * np.sum(diff * inv_dist6[:, :, np.newaxis], axis=1)
        return grad.flatten()

    try:
        res = minimize(energy, c0.flatten(), jac=energy_grad, method='L-BFGS-B', 
                       bounds=[(-0.5, 0.5)] * (N * 3), options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
        c_opt = res.x.reshape(N, 3)
    except Exception:
        c_opt = c0

    c_opt = np.clip(c_opt, -0.5, 0.5)
    
    # Final jiggle to maximize min distance
    # Use a simple gradient step on the min distance approximation
    for _ in range(50):
        if time.time() > deadline:
            break
        diff = c_opt[:, np.newaxis, :] - c_opt[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Calculate gradient of min dist (approximate by summing repulsions from closest neighbors)
        # Or just use the L-BFGS result which should be good.
        # Let's do a few steps of "push apart" for pairs closer than average
        avg_dist = np.mean(dist[dist < np.inf])
        close_mask = dist < avg_dist * 1.1
        np.fill_diagonal(close_mask, False)
        
        # Move closer pairs apart
        for i in range(N):
            for j in range(i+1, N):
                if close_mask[i, j]:
                    d = dist[i, j]
                    if d > 1e-9:
                        push = 0.01 * (1.0 / (d**2))
                        dir_vec = (c_opt[i] - c_opt[j]) / d
                        c_opt[i] += dir_vec * push
                        c_opt[j] -= dir_vec * push
        
        c_opt = np.clip(c_opt, -0.5, 0.5)

    return c_opt
