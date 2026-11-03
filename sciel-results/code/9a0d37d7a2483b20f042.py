"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _objective_and_grad(pts):
    """
    Objective: Variance of pairwise angular distances.
    Gradient: Analytical.
    """
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    pts_norm = pts / norms
    
    G = pts_norm @ pts_norm.T
    i, j = np.triu_indices(N, 1)
    dots = G[i, j]
    dots = np.clip(dots, -0.9999999, 0.9999999)
    
    angles = np.arccos(dots)
    mean_angle = np.mean(angles)
    
    # Objective
    var = np.var(angles)
    
    # Gradient calculation
    denom = np.sqrt(1.0 - dots * dots)
    denom = np.where(denom < 1e-12, 1e-12, denom)
    w = -1.0 / denom
    
    M = len(angles)
    grad_theta = (2.0 / M) * (angles - mean_angle)
    
    grad_pts = np.zeros_like(pts_norm)
    
    # Accumulate gradients
    # grad_pts[i] += grad_theta[k] * w[k] * pts_norm[j]
    # grad_pts[j] += grad_theta[k] * w[k] * pts_norm[i]
    
    # Using np.add.at for vectorized accumulation might be faster
    # But simple loop is fine for N=130
    
    coeff = grad_theta * w
    
    for k in range(M):
        idx_i = i[k]
        idx_j = j[k]
        c = coeff[k]
        grad_pts[idx_i] += c * pts_norm[idx_j]
        grad_pts[idx_j] += c * pts_norm[idx_i]
        
    return var, grad_pts.flatten()


def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    x0 = pts.flatten()
    
    start_time = time.time()
    
    try:
        result = minimize(
            _objective_and_grad, 
            x0, 
            method='L-BFGS-B', 
            jac=True,
            options={
                'maxiter': 200, 
                'maxfun': 5000,
                'ftol': 1e-10,
                'gtol': 1e-8
            }
        )
        
        x_opt = result.x
        pts_opt = x_opt.reshape(N, 3)
        
        # Final projection
        norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        pts_opt = pts_opt / norms
        
        elapsed = time.time() - start_time
        if elapsed > 110:
            return pts
            
        return pts_opt

    except Exception:
        return pts

# EVOLVE-BLOCK-END
