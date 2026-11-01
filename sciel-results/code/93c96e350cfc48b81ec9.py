import numpy as np
import time
from scipy.optimize import minimize

N = 300
R_KNOWN = 0.0531542991469

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin

    # 1. Generate initial hexagonal lattice configuration
    def get_hex_lattice():
        best_points = None
        best_s = float('inf')
        # Search for a spacing that yields at least N points in the unit disk
        for s in np.linspace(0.09, 0.12, 30):
            k = int(1.0 / s) + 2
            rows = [((i + 0.5 * (j % 2)) * s, j * s * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            mask = radius <= 1.0
            points = points[mask]
            if len(points) >= N:
                idx = np.argsort(radius)
                points = points[idx[:N]]
                if s < best_s:
                    best_points = points
                    best_s = s
        return best_points

    init_points = get_hex_lattice()
    if init_points is None:
        init_points = np.random.rand(N, 2) * 2 - 1
        init_points *= 0.8

    # Target radius
    r_target = R_KNOWN * 0.9999 

    def penalty(x):
        c = x.reshape(N, 2)
        dist_center = np.linalg.norm(c, axis=1)
        boundary_violation = np.maximum(0, dist_center - (1.0 - r_target))
        penalty_boundary = np.sum(boundary_violation**2)
        
        dx = c[:, np.newaxis, 0] - c[np.newaxis, :, 0]
        dy = c[:, np.newaxis, 1] - c[np.newaxis, :, 1]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, np.inf)
        overlap_violation = np.maximum(0, 2 * r_target - d)
        penalty_overlap = np.sum(overlap_violation**2)
        
        return penalty_boundary + penalty_overlap

    def gradient(x):
        c = x.reshape(N, 2)
        grad = np.zeros_like(c)
        
        dist_center = np.linalg.norm(c, axis=1)
        mask_b = dist_center > (1.0 - r_target)
        if np.any(mask_b):
            coeff = 2 * (dist_center[mask_b] - (1.0 - r_target)) / dist_center[mask_b]
            grad[mask_b] += coeff[:, np.newaxis] * c[mask_b]
            
        dx = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(dx, axis=2)
        np.fill_diagonal(d, 1.0)
        
        mask_o = d < 2 * r_target
        force_mag = np.zeros_like(d)
        force_mag[mask_o] = 2 * (2 * r_target - d[mask_o]) / d[mask_o]
        
        grad += np.sum(force_mag[:, :, np.newaxis] * dx, axis=1)
        
        return grad.flatten()

    bounds = [(-1.0, 1.0)] * (2 * N)
    
    best_x = init_points.flatten()
    best_val = penalty(best_x)
    
    # Run optimization with a few restarts
    for i in range(5):
        if time.time() - start_time > time_limit - 10:
            break
            
        if i == 0:
            x0 = init_points.flatten()
        else:
            # Perturb the best solution found so far
            x0 = best_x + np.random.normal(0, 0.01, best_x.shape)
            
        try:
            res = minimize(penalty, x0, jac=gradient, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_val:
                best_x = res.x
                best_val = res.fun
        except Exception:
            pass

    centers = best_x.reshape(N, 2)
    
    # Final safety check: ensure inside unit circle
    dist = np.linalg.norm(centers, axis=1)
    mask = dist > 1.0
    if np.any(mask):
        centers[mask] /= dist[mask, np.newaxis]
        
    return centers
