import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100

def _compute_energy_and_grad_flat(flat_pts, C_penalty=1e6):
    """
    Compute energy and gradient for flat array of shape (3*N,).
    Includes penalty for sphere constraint.
    """
    pts = flat_pts.reshape(N, 3)
    
    # Pairwise differences: (N, N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    
    # Squared distances: (N, N)
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero on diagonal
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    
    # Distances
    # We need 1 / dist_sq for gradient, and log(dist) for energy.
    # dist = sqrt(dist_sq)
    # log(1/dist) = -0.5 * log(dist_sq)
    
    # Energy
    # Sum over i < j of -log(dist_ij)
    # dist_sq[i,j] is symmetric.
    # We sum over upper triangle.
    iu = np.triu_indices(N, k=1)
    d_sq_upper = dist_sq[iu]
    # Clamp to avoid log(0) if points coincide (shouldn't happen with penalty, but safe)
    d_sq_upper = np.maximum(d_sq_upper, 1e-30)
    energy = -0.5 * np.sum(np.log(d_sq_upper))
    
    # Gradient of Energy
    # grad_i = sum_{j != i} (x_j - x_i) / dist_sq_ij
    # Note: diff[i,j] = x_i - x_j. So x_j - x_i = -diff[i,j].
    # term[i,j] = -diff[i,j] / dist_sq[i,j]
    # We need to sum over j for each i.
    
    # Create inverse dist sq matrix, zero on diagonal
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    
    # term[i,j,k] = -diff[i,j,k] * inv_dist_sq[i,j]
    grad_e = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Penalty Term
    # P = C * sum (||x_i||^2 - 1)^2
    # dP/dx_i = 4 * C * x_i * (||x_i||^2 - 1)
    norms_sq = np.sum(pts * pts, axis=1)
    grad_p = 4.0 * C_penalty * pts * (norms_sq - 1.0)[:, np.newaxis]
    
    total_grad = grad_e + grad_p
    
    return energy + C_penalty * np.sum((norms_sq - 1.0)**2), total_grad.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Flatten for scipy
    x0 = init_pts.flatten()
    
    # Optimize
    # L-BFGS-B is fast and handles the smooth penalty well
    # We use a high penalty to keep points on the sphere
    C_pen = 1e6
    
    def objective(x):
        val, _ = _compute_energy_and_grad_flat(x, C_pen)
        return val

    def jac(x):
        _, g = _compute_energy_and_grad_flat(x, C_pen)
        return g

    best_x = x0
    best_val = objective(best_x)
    
    # Run optimizer
    try:
        res = minimize(objective, x0, jac=jac, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.success or res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return best_x.reshape(N, 3)
    
    # Try a few random restarts if time permits
    np.random.seed(42)
    for restart in range(10):
        if time.time() > deadline - 10.0:
            break
            
        # Random init
        pts_r = np.random.randn(N, 3)
        norms = np.linalg.norm(pts_r, axis=1, keepdims=True)
        pts_r = pts_r / norms
        x_r = pts_r.flatten()
        
        try:
            res_r = minimize(objective, x_r, jac=jac, method='L-BFGS-B', 
                             options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            if res_r.fun < best_val:
                best_val = res_r.fun
                best_x = res_r.x
        except Exception:
            continue
            
    # Final projection to ensure exact unit sphere (remove any drift from penalty)
    final_pts = best_x.reshape(N, 3)
    final_norms = np.linalg.norm(final_pts, axis=1, keepdims=True)
    final_pts = final_pts / final_norms
    
    return final_pts
