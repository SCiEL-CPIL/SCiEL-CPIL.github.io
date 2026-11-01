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
    
    P = 20.0  # Power for approximation

    def objective_and_grad(x):
        pts = x.reshape(N, 3)
        # Pairwise part
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Avoid division by zero
        dists_safe = np.where(dists < 1e-8, 1e-8, dists)
        
        # Objective value
        sum_pair = np.sum(1.0 / (dists_safe ** P))
        
        # Gradient of pairwise part
        # grad_i = -P * sum_j (d_ij^(-P-2) * (x_i - x_j))
        # dists_safe^(-P-2) * diff
        # diff[i,j] = x_i - x_j
        # We need sum over j for each i
        # Weight matrix W = dists_safe^(-P-2)
        W = dists_safe ** (-P - 2.0)
        np.fill_diagonal(W, 0.0)
        
        # grad_pair[i] = -P * sum_j W[i,j] * (x_i - x_j)
        # sum_j W[i,j] * x_i = x_i * sum_j W[i,j]
        # sum_j W[i,j] * x_j = (W @ pts)[i]
        row_sums = np.sum(W, axis=1)
        grad_pair = -P * (row_sums[:, np.newaxis] * pts - (W @ pts))
        
        # Boundary part
        # d_b,i = min_k (0.5 - |x_ik|)
        abs_pts = np.abs(pts)
        # dist to boundary
        b_dists = 0.5 - abs_pts
        min_idx = np.argmin(b_dists, axis=1)
        min_b_vals = np.min(b_dists, axis=1)
        
        min_b_safe = np.where(min_b_vals < 1e-8, 1e-8, min_b_vals)
        sum_bound = np.sum(1.0 / (min_b_safe ** P))
        
        # Gradient of boundary part
        # grad_i = -P * min_b_i^(-P-1) * grad(min_b_i)
        # grad(min_b_i) is non-zero only for the axis k that is the min.
        # d/dx (0.5 - |x|) = -sign(x)
        grad_bound = np.zeros_like(pts)
        for i in range(N):
            k = min_idx[i]
            # sign of pts[i,k]
            s = np.sign(pts[i, k])
            if s == 0: s = 1.0
            grad_bound[i, k] = -P * (min_b_safe[i] ** (-P - 1.0)) * (-s)
            # Wait: d/dx (0.5-|x|) = -sign(x).
            # d/dx ( (0.5-|x|)^-P ) = -P (0.5-|x|)^(-P-1) * (-sign(x)) = P (0.5-|x|)^(-P-1) * sign(x)
            # So grad_bound[i, k] = P * min_b_safe[i]**(-P-1) * s
            
        total_grad = grad_pair + grad_bound
        total_obj = sum_pair + sum_bound
        
        return total_obj, total_grad.flatten()

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

    starts = [
        init_grid(),
        init_grid() + np.random.normal(0, 0.02, size=(N*3,)),
        init_random(),
        init_random()
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
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-9})
            if res.fun < 1e10:
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
