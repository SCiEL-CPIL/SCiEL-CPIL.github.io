"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _log_energy_and_grad(pts):
    """Compute log energy sum_{i<j} log(1/|x_i - x_j|) and its gradient w.r.t. pts.
    Energy = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    Gradient of log(d_ij) w.r.t. x_i = (x_i - x_j) / d_ij^2
    So gradient of -log(d_ij) w.r.t. x_i = -(x_i - x_j) / d_ij^2
    """
    n = pts.shape[0]
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n,n,3)
    dist2 = np.sum(diff * diff, axis=2)  # (n,n)
    np.fill_diagonal(dist2, 1.0)  # avoid div by zero
    dist = np.sqrt(dist2)
    
    # Energy: sum over i<j of log(1/d) = -sum_{i<j} log(d)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    d_upper = dist[iu]
    energy = -np.sum(np.log(d_upper))
    
    # Gradient: for each pair (i,j), i<j:
    # d/dx_i [-log(d_ij)] = -(x_i - x_j) / d_ij^2
    # d/dx_j [-log(d_ij)] = +(x_i - x_j) / d_ij^2
    grad = np.zeros_like(pts)
    # diff[i,j] = x_i - x_j
    diff_upper = diff[iu]  # (n(n-1)/2, 3)
    d2_upper = dist2[iu]  # (n(n-1)/2,)
    coeff = 1.0 / d2_upper  # (n(n-1)/2,)
    
    # For each pair, add -coeff * diff to grad[i], +coeff * diff to grad[j]
    ii = iu[0]
    jj = iu[1]
    np.add.at(grad, ii, -coeff[:, np.newaxis] * diff_upper)
    np.add.at(grad, jj, coeff[:, np.newaxis] * diff_upper)
    
    return energy, grad


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    best_pts = pts.copy()
    best_energy = _log_energy_and_grad(pts)[0]
    
    # Try multiple random restarts with gradient descent on sphere
    rng = np.random.default_rng(42)
    
    for restart in range(20):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb initial points
        if restart == 0:
            p = pts.copy()
        else:
            p = pts + 0.1 * rng.standard_normal((N, 3))
            p = p / np.linalg.norm(p, axis=1, keepdims=True)
        
        # Gradient descent with projection onto sphere
        step_size = 0.01
        for step in range(500):
            if time.time() > deadline - 5.0:
                break
            
            e, g = _log_energy_and_grad(p)
            
            # Project gradient to tangent space
            p_norm = p / np.linalg.norm(p, axis=1, keepdims=True)
            g_tangent = g - np.sum(g * p_norm, axis=1, keepdims=True) * p_norm
            
            # Adaptive step
            p_new = p - step_size * g_tangent
            p_new = p_new / np.linalg.norm(p_new, axis=1, keepdims=True)
            
            e_new, _ = _log_energy_and_grad(p_new)
            if e_new < e:
                p = p_new
                # Reduce step slowly
                step_size = max(0.001, step_size * 0.999)
            else:
                step_size *= 0.5
        
        e_final, _ = _log_energy_and_grad(p)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = p.copy()
    
    # Final: try L-BFGS-B on flattened coordinates with sphere constraint via penalty
    if time.time() < deadline - 15.0:
        def sphere_energy_flat(x):
            p = x.reshape(N, 3)
            p = p / np.linalg.norm(p, axis=1, keepdims=True)
            e, _ = _log_energy_and_grad(p)
            return e
        
        def sphere_grad_flat(x):
            p = x.reshape(N, 3)
            p = p / np.linalg.norm(p, axis=1, keepdims=True)
            e, g = _log_energy_and_grad(p)
            # Project gradient
            g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
            return g_tangent.flatten()
        
        x0 = best_pts.flatten()
        try:
            res = minimize(sphere_energy_flat, x0, jac=sphere_grad_flat, 
                          method='L-BFGS-B', options={'maxiter': 1000, 'gtol': 1e-12})
            p_opt = res.x.reshape(N, 3)
            p_opt = p_opt / np.linalg.norm(p_opt, axis=1, keepdims=True)
            e_opt, _ = _log_energy_and_grad(p_opt)
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = p_opt
        except Exception:
            pass
    
    # Ensure valid output
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
# EVOLVE-BLOCK-END
