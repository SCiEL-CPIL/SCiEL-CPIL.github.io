"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def _energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|)
    and its gradient w.r.t. the Cartesian coordinates.
    E = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = - sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist2 = np.sum(diff * diff, axis=2)  # (n, n)
    # Avoid division by zero on the diagonal
    np.fill_diagonal(dist2, 1.0)
    
    dist = np.sqrt(dist2)
    
    # Energy: sum over i<j of log(1/dist[i,j]) = -sum over i<j log(dist[i,j])
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: dE/dx_i = - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    weight = np.zeros_like(dist2)
    np.fill_diagonal(weight, 0.0)
    weight = 1.0 / dist2
    
    grad = -np.einsum('ij,ijk->ik', weight, diff)
    
    return energy, grad

def _retract(pts):
    """Project points back onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _lbfgs_sphere(pts, deadline, maxiter=5000):
    """Run L-BFGS-B on the sphere with time guard."""
    x0 = pts.flatten()
    best_x = x0.copy()
    best_f, _ = _energy_and_grad(_retract(pts))
    
    def obj_and_grad(x_flat):
        p = _retract(x_flat.reshape(N, 3))
        f, g = _energy_and_grad(p)
        # Project gradient onto tangent space for proper manifold optimization
        proj = np.einsum('ij,ij->i', g, p)
        g_tangent = g - proj[:, np.newaxis] * p
        return f, g_tangent.flatten()
    
    def callback(xk):
        nonlocal best_x, best_f
        if time.time() > deadline:
            return True
        f_k, _ = obj_and_grad(xk)
        if f_k < best_f:
            best_f = f_k
            best_x = xk.copy()
        return False
    
    try:
        res = minimize(
            obj_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': maxiter,
                'ftol': 1e-14,
                'gtol': 1e-12,
                'maxfun': maxiter * 2
            },
            callback=callback
        )
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x
    except Exception:
        pass
    
    return _retract(best_x.reshape(N, 3)), best_f

def _polish(pts, deadline, n_steps=200):
    """Gradient descent polish with adaptive step size on the sphere."""
    f, g = _energy_and_grad(pts)
    lr = 0.01
    
    for _ in range(n_steps):
        if time.time() > deadline:
            break
        
        # Project gradient onto tangent space
        proj = np.einsum('ij,ij->i', g, pts)
        g_tangent = g - proj[:, np.newaxis] * pts
        
        new_pts = _retract(pts - lr * g_tangent)
        new_f, new_g = _energy_and_grad(new_pts)
        
        if new_f < f:
            pts = new_pts
            f = new_f
            g = new_g
            lr *= 0.995
        else:
            lr *= 0.5
            if lr < 1e-12:
                break
    
    return pts, f

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initial configuration: Golden-angle spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts_init = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Run L-BFGS-B
    pts_opt, best_f = _lbfgs_sphere(pts_init, deadline)
    
    # Polish if time permits
    if time.time() < deadline - 3.0:
        pts_opt, best_f = _polish(pts_opt, deadline)
    
    # Final ultra-polish with very small steps
    if time.time() < deadline - 1.0:
        f, g = _energy_and_grad(pts_opt)
        for _ in range(50):
            if time.time() > deadline - 0.5:
                break
            proj = np.einsum('ij,ij->i', g, pts_opt)
            g_t = g - proj[:, np.newaxis] * pts_opt
            new_pts = _retract(pts_opt - 1e-4 * g_t)
            new_f, new_g = _energy_and_grad(new_pts)
            if new_f < f:
                pts_opt = new_pts
                f = new_f
                g = new_g
    
    return pts_opt
# EVOLVE-BLOCK-END
