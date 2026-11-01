"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = -1.0
    best_points = None
    
    def calculate_score(points):
        if points is None:
            return 0.0
        pts = points.reshape(N, 3)
        if np.any(pts < -0.5) or np.any(pts > 0.5):
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
            
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        boundary_dists = 0.5 - np.abs(pts)
        min_boundary = np.min(boundary_dists)
        
        r = min(min_pair / 2.0, min_boundary)
        return r

    def init_grid():
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].flatten()

    def init_random():
        return np.random.uniform(-0.4, 0.4, size=(N, 3)).flatten()

    def init_fibonacci():
        # Fibonacci sphere-like distribution in a cube
        pts = np.zeros((N, 3))
        for i in range(N):
            t = i / (N - 1) if N > 1 else 0.5
            # Use golden ratio for better distribution
            golden = (1 + 5**0.5) / 2
            angle = 2 * np.pi * i / golden
            radius = np.sqrt((i + 0.5) / N)
            z = 2 * (i + 0.5) / N - 1
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            # Scale to cube
            scale = 0.45 / max(abs(x), abs(y), abs(z), 1e-10)
            pts[i] = [x * scale, y * scale, z * 0.45]
        return pts.flatten()

    def objective_and_grad(x):
        pts = x.reshape(N, 3)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Use a soft minimum approximation
        P = 30.0
        dists_safe = np.maximum(dists, 1e-8)
        
        # Objective: minimize sum of (1/d)^P
        sum_pair = np.sum(dists_safe ** (-P))
        
        # Gradient
        W = dists_safe ** (-P - 2.0)
        np.fill_diagonal(W, 0.0)
        
        row_sums = np.sum(W, axis=1)
        grad_pair = -P * (row_sums[:, np.newaxis] * pts - (W @ pts))
        
        # Boundary constraint
        b_dists = 0.5 - np.abs(pts)
        min_idx = np.argmin(b_dists, axis=1)
        min_b_vals = np.min(b_dists, axis=1)
        
        min_b_safe = np.maximum(min_b_vals, 1e-8)
        sum_bound = np.sum(min_b_safe ** (-P))
        
        # Boundary gradient
        grad_bound = np.zeros_like(pts)
        for i in range(N):
            k = min_idx[i]
            s = np.sign(pts[i, k])
            if s == 0:
                s = 1.0
            # d/dx_k (0.5 - |x_k|)^(-P) = P * (0.5 - |x_k|)^(-P-1) * sign(x_k)
            grad_bound[i, k] = P * (min_b_safe[i] ** (-P - 1.0)) * s
        
        total_grad = grad_pair + grad_bound
        total_obj = sum_pair + sum_bound
        
        return total_obj, total_grad.flatten()

    starts = [
        init_grid(),
        init_grid() + np.random.normal(0, 0.015, size=(N*3,)),
        init_grid() + np.random.normal(0, 0.03, size=(N*3,)),
        init_random(),
        init_random() + np.random.normal(0, 0.02, size=(N*3,)),
        init_fibonacci(),
        init_fibonacci() + np.random.normal(0, 0.01, size=(N*3,))
    ]
    
    # Clip starts to valid range
    for i in range(len(starts)):
        starts[i] = np.clip(starts[i], -0.49, 0.49)

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    for i, x0 in enumerate(starts):
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective_and_grad, x0, method='L-BFGS-B', bounds=bounds, 
                           jac=True,
                           options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
            if res.fun < 1e12:
                pts = res.x.reshape(N, 3)
                pts = np.clip(pts, -0.5, 0.5)
                score = calculate_score(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
        except Exception:
            continue

    if best_points is None:
        best_points = init_grid().reshape(N, 3)
        
    return best_points
# EVOLVE-BLOCK-END
