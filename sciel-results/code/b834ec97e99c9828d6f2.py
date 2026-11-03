"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _log_energy(pts):
    """Compute sum_{i<j} log(1/|x_i - x_j|) using math.fsum for precision."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dists[mask]
    if np.any(d < 1e-9):
        return float('inf')
    terms = -np.log(d)
    return math.fsum(terms.tolist())

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _fibonacci_spiral():
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(seed):
    """Generate random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    return _normalize(pts)

def _energy_and_grad(pts):
    """Compute energy and gradient of log-energy on sphere."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    
    inv_d2 = 1.0 / (dists * dists)
    inv_d2[np.diag_indices(n)] = 0.0
    
    # Gradient of log(1/d) = -log(d) is (p_j - p_i)/d_ij^2 for each pair
    # So grad_i = sum_j (p_j - p_i) / d_ij^2 = -sum_j (p_i - p_j) / d_ij^2
    grad = -np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    # Project to tangent space
    dots = np.sum(grad * pts, axis=1)
    proj_grad = grad - dots[:, np.newaxis] * pts
    
    # Energy
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dists[mask]
    if np.any(d < 1e-9):
        return float('inf'), proj_grad
    energy = math.fsum((-np.log(d)).tolist())
    
    return energy, proj_grad

def _langevin_dynamics(pts, dt=0.01, temp=0.1, n_steps=1000, seed=42):
    """
    Velocity Verlet Langevin dynamics on the sphere with adaptive temperature.
    """
    rng = np.random.RandomState(seed)
    x = pts.copy()
    v = np.zeros_like(x)
    
    best_x = x.copy()
    best_e, _ = _energy_and_grad(x)
    
    # Compute initial forces
    _, grad = _energy_and_grad(x)
    # Project to tangent
    dots = np.sum(grad * x, axis=1)
    F = -(grad - dots[:, np.newaxis] * x)  # negative gradient = force
    
    for step in range(n_steps):
        # Adaptive temperature: decrease over time
        current_temp = temp * np.exp(-0.001 * step)
        
        # Velocity Verlet step 1: update position by half step with current velocity
        x_half = x + v * (dt / 2.0) + 0.5 * F * (dt / 2.0) ** 2
        x_half = _normalize(x_half)
        
        # Compute force at new position
        _, grad_new = _energy_and_grad(x_half)
        dots_new = np.sum(grad_new * x_half, axis=1)
        F_new = -(grad_new - dots_new[:, np.newaxis] * x_half)
        
        # Add Langevin noise and friction
        noise = rng.randn(*x_half.shape) * np.sqrt(2.0 * current_temp * dt)
        friction = np.exp(-dt)
        v_new = (v + 0.5 * (F + F_new) * dt) * friction + noise
        
        # Update position
        x_new = x_half + v_new * (dt / 2.0)
        x_new = _normalize(x_new)
        
        # Check energy
        e_new, _ = _energy_and_grad(x_new)
        if e_new < best_e and np.isfinite(e_new):
            best_e = e_new
            best_x = x_new.copy()
        
        x, v, F = x_new, v_new, F_new
        
    return best_x, best_e

def _lbfgs_optimize(pts, maxiter=2000):
    """Polish with L-BFGS in spherical coordinates."""
    # Convert to spherical coordinates for optimization
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    theta = np.arctan2(y, x)
    phi = np.arccos(np.clip(z, -1.0, 1.0))
    
    def obj(params):
        th = params[:N]
        ph = params[N:2*N]
        px = np.sin(ph) * np.cos(th)
        py = np.sin(ph) * np.sin(th)
        pz = np.cos(ph)
        p = np.column_stack([px, py, pz])
        e, _ = _energy_and_grad(p)
        return e
    
    params0 = np.concatenate([theta, phi])
    result = minimize(obj, params0, method='L-BFGS-B', 
                      options={'maxiter': maxiter, 'ftol': 1e-18, 'gtol': 1e-15, 'maxfun': 50000})
    
    th = result.x[:N]
    ph = result.x[N:2*N]
    px = np.sin(ph) * np.cos(th)
    py = np.sin(ph) * np.sin(th)
    pz = np.cos(ph)
    p = np.column_stack([px, py, pz])
    return p, result.fun

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_pts = _fibonacci_spiral()
    best_energy = _log_energy(best_pts)
    
    # Multiple restarts with different strategies
    n_restarts = 25
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Different initialization strategies
        if restart == 0:
            init_pts = _fibonacci_spiral()
        elif restart % 3 == 1:
            init_pts = _random_sphere(restart * 12345)
        elif restart % 3 == 2:
            # Perturbed Fibonacci
            init_pts = _fibonacci_spiral() + 0.1 * np.random.randn(N, 3)
            init_pts = _normalize(init_pts)
        else:
            # Golden ratio variant
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i * (1.0 + 0.01 * restart)
            r = np.sqrt(1.0 - z * z)
            init_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # Phase 1: Langevin dynamics for broad exploration
        if time.time() < deadline - 25:
            pts_l, e_l = _langevin_dynamics(init_pts, dt=0.005, temp=0.05, 
                                           n_steps=500, seed=restart * 777)
            if e_l < best_energy:
                best_energy = e_l
                best_pts = pts_l.copy()
        
        # Phase 2: L-BFGS polish
        if time.time() < deadline - 10:
            pts_p, e_p = _lbfgs_optimize(best_pts, maxiter=1000)
            if e_p < best_energy and np.isfinite(e_p):
                best_energy = e_p
                best_pts = pts_p.copy()
    
    # Final precise energy check
    final_e = _log_energy(best_pts)
    if np.isfinite(final_e):
        return best_pts
    else:
        return _fibonacci_spiral()
# EVOLVE-BLOCK-END
