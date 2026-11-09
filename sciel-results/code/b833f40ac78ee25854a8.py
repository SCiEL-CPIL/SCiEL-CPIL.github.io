"""Optimized logarithmic energy minimization for 150 points on the sphere."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
TIME_LIMIT = 140.0  # safety margin before 150s kill


def _energy_and_grad(pts):
    """Compute log energy sum_{i<j} log(1/|x_i - x_j|) and its gradient w.r.t. pts.
    Energy = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    Gradient of -log(d_ij) w.r.t. x_i = -(x_i - x_j) / d_ij^2
    """
    n = pts.shape[0]
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n,n,3)
    dist2 = np.sum(diff * diff, axis=2)  # (n,n)
    np.fill_diagonal(dist2, 1.0)  # avoid div by zero
    dist = np.sqrt(dist2)
    
    # Energy: sum over i<j of log(1/d) = -sum_{i<j} log(d)
    iu = np.triu_indices(n, k=1)
    d_upper = dist[iu]
    energy = -np.sum(np.log(d_upper))
    
    # Gradient: for each pair (i,j), i<j:
    # d/dx_i [-log(d_ij)] = -(x_i - x_j) / d_ij^2
    # d/dx_j [-log(d_ij)] = +(x_i - x_j) / d_ij^2
    grad = np.zeros_like(pts)
    diff_upper = diff[iu]  # (n(n-1)/2, 3)
    d2_upper = dist2[iu]  # (n(n-1)/2,)
    coeff = 1.0 / d2_upper  # (n(n-1)/2,)
    
    ii = iu[0]
    jj = iu[1]
    np.add.at(grad, ii, -coeff[:, np.newaxis] * diff_upper)
    np.add.at(grad, jj, coeff[:, np.newaxis] * diff_upper)
    
    return energy, grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _fibonacci_spiral(n):
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(pts)


def _random_initialization(n, rng):
    """Generate random initial points on the sphere."""
    pts = rng.standard_normal((n, 3))
    return _project_to_sphere(pts)


def _gradient_descent_sphere(pts, max_iter=2000, tol=1e-12, deadline=None):
    """Gradient descent on the sphere with adaptive step size."""
    n = pts.shape[0]
    p = pts.copy()
    step_size = 0.01
    best_e = _energy_and_grad(p)[0]
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        e, g = _energy_and_grad(p)
        
        # Project gradient to tangent space
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        
        # Adaptive step
        p_new = p - step_size * g_tangent
        p_new = _project_to_sphere(p_new)
        
        e_new, _ = _energy_and_grad(p_new)
        if e_new < e:
            p = p_new
            step_size = min(0.1, step_size * 1.05)
        else:
            step_size *= 0.5
        
        if step_size < 1e-10:
            break
    
    return p, _energy_and_grad(p)[0]


def _lbfgs_optimize(pts, deadline):
    """Optimize using L-BFGS-B with sphere constraint via normalization."""
    n = pts.shape[0]
    
    def energy_flat(x):
        p = x.reshape(n, 3)
        p = _project_to_sphere(p)
        e, _ = _energy_and_grad(p)
        return e
    
    def grad_flat(x):
        p = x.reshape(n, 3)
        p = _project_to_sphere(p)
        e, g = _energy_and_grad(p)
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g_tangent.flatten()
    
    x0 = pts.flatten()
    try:
        res = minimize(energy_flat, x0, jac=grad_flat, 
                      method='L-BFGS-B', options={'maxiter': 500, 'gtol': 1e-12})
        p_opt = _project_to_sphere(res.x.reshape(n, 3))
        e_opt = _energy_and_grad(p_opt)[0]
        return p_opt, e_opt
    except Exception:
        return pts, _energy_and_grad(pts)[0]


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Initial: Fibonacci spiral
    pts = _fibonacci_spiral(N)
    best_pts = pts.copy()
    best_energy = _energy_and_grad(pts)[0]
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts with gradient descent
    for restart in range(15):
        if time.time() > deadline - 15.0:
            break
        
        if restart == 0:
            p = pts.copy()
        else:
            p = _random_initialization(N, rng)
        
        # Gradient descent
        p_gd, e_gd = _gradient_descent_sphere(p, max_iter=1500, deadline=deadline - 10.0)
        
        # L-BFGS refinement
        if time.time() < deadline - 5.0:
            p_lbfgs, e_lbfgs = _lbfgs_optimize(p_gd, deadline)
            if e_lbfgs < e_gd:
                p_gd, e_gd = p_lbfgs, e_lbfgs
        
        if e_gd < best_energy:
            best_energy = e_gd
            best_pts = p_gd.copy()
    
    # Final L-BFGS on best result
    if time.time() < deadline - 2.0:
        p_final, e_final = _lbfgs_optimize(best_pts, deadline)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = p_final
    
    return _project_to_sphere(best_pts)
