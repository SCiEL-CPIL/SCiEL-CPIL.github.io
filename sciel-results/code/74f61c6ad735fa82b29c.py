"""Rewritten program: L-BFGS-B on spherical coordinates for log-energy minimization."""
import numpy as np
import math
import time
from scipy.optimize import minimize, Bounds

N = 150
TOTAL_TIME = 150.0
SAFETY_MARGIN = 5.0

def _energy_and_grad_spherical(theta, phi):
    """Compute energy and gradient in spherical coordinates (theta, phi).
    theta: polar angle [0, pi], phi: azimuthal angle [0, 2*pi]
    Returns (energy, grad_theta, grad_phi).
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    
    pts = np.column_stack([sin_t * cos_p, sin_t * sin_p, cos_t])
    
    # Pairwise squared distances: ||xi - xj||^2 = 2 - 2*xi.xj
    dots = pts @ pts.T
    sq_dists = 2.0 - 2.0 * dots
    np.fill_diagonal(sq_dists, 1.0)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    iu = np.triu_indices(N, k=1)
    sq_d_iu = sq_dists[iu]
    
    # Energy = sum log(1/d) = -0.5 * sum log(d^2)
    log_sq = np.log(sq_d_iu)
    energy = -0.5 * math.fsum(log_sq.tolist())
    
    # Gradient w.r.t. points (tangent-projected)
    # dE/dxi = sum_{j != i} (xi - xj) / ||xi - xj||^2
    weights = 1.0 / sq_dists
    np.fill_diagonal(weights, 0.0)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    grad_pts = np.sum(diffs * weights[..., np.newaxis], axis=1)
    
    # Project to tangent space: g - (g.x) x
    gdotx = np.sum(grad_pts * pts, axis=1)
    grad_tangent = grad_pts - gdotx[:, np.newaxis] * pts
    
    # Chain rule: dE/dtheta = grad_tangent . dxi/dtheta
    # dxi/dtheta = [cos_t*cos_p, cos_t*sin_p, -sin_t]
    dxi_dtheta = np.column_stack([cos_t * cos_p, cos_t * sin_p, -sin_t])
    grad_theta = np.sum(grad_tangent * dxi_dtheta, axis=1)
    
    # dE/dphi = grad_tangent . dxi/dphi
    # dxi/dphi = [-sin_p, cos_p, 0]
    dxi_dphi = np.column_stack([-sin_p, cos_p, np.zeros(N)])
    grad_phi = np.sum(grad_tangent * dxi_dphi, axis=1)
    
    return energy, np.concatenate([grad_theta, grad_phi])

def _energy_fast(pts):
    """Compute logarithmic energy for points on sphere."""
    sq_dists = 2.0 - 2.0 * (pts @ pts.T)
    np.fill_diagonal(sq_dists, 1.0)
    sq_dists = np.maximum(sq_dists, 1e-15)
    iu = np.triu_indices(N, k=1)
    log_dists_sq = np.log(sq_dists[iu])
    return -0.5 * math.fsum(log_dists_sq.tolist())

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _points_to_spherical(pts):
    """Convert 3D points on sphere to (theta, phi)."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    phi = np.mod(phi, 2.0 * np.pi)
    return theta, phi

def _optimize_spherical(theta0, phi0, time_limit, start_time, maxiter=5000):
    """Run L-BFGS-B on spherical coordinates."""
    x0 = np.concatenate([theta0, phi0])
    
    def fun_and_grad(x):
        if time.time() - start_time > time_limit:
            raise RuntimeError("timeout")
        t = x[:N]
        p = x[N:]
        e, g = _energy_and_grad_spherical(t, p)
        return e, g
    
    bounds = [(0.0, np.pi)] * N + [(0.0, 2.0 * np.pi)] * N
    
    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                       bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-12,
                                               'gtol': 1e-10, 'maxfun': 10000})
    except RuntimeError:
        # Timeout - return current best if available
        pass
    except Exception:
        pass
    
    # Extract best point from result
    if 'res' in dir() and res.x is not None:
        t = np.clip(res.x[:N], 1e-10, np.pi - 1e-10)
        p = res.x[N:]
        cos_t = np.cos(t)
        sin_t = np.sin(t)
        pts = np.column_stack([sin_t * np.cos(p), sin_t * np.sin(p), cos_t])
        return _project_to_sphere(pts), _energy_fast(pts)
    else:
        return None, float('inf')

def _refine_direct(pts, time_limit, start_time, maxiter=3000):
    """Direct refinement using L-BFGS-B with sphere projection via penalty."""
    # Use a different approach: optimize in 3D with projection
    # Actually, let's use a simpler gradient descent with momentum on the sphere
    
    pts = _project_to_sphere(pts.copy())
    best_energy = _energy_fast(pts)
    best_pts = pts.copy()
    
    lr = 0.01
    momentum = 0.9
    velocity = np.zeros_like(pts)
    
    for i in range(maxiter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute gradient
        sq_dists = 2.0 - 2.0 * (pts @ pts.T)
        np.fill_diagonal(sq_dists, 1.0)
        sq_dists = np.maximum(sq_dists, 1e-15)
        weights = 1.0 / sq_dists
        np.fill_diagonal(weights, 0.0)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        grad = np.sum(diffs * weights[..., np.newaxis], axis=1)
        gdotx = np.sum(grad * pts, axis=1)
        grad = grad - gdotx[:, np.newaxis] * pts
        
        # Momentum update
        velocity = momentum * velocity - lr * grad
        pts_new = pts + velocity
        pts_new = _project_to_sphere(pts_new)
        
        energy = _energy_fast(pts_new)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts_new.copy()
        
        pts = pts_new
        # Decay learning rate
        lr *= 0.9995
    
    return best_pts, best_energy

def construct():
    deadline = time.time() + TOTAL_TIME - SAFETY_MARGIN
    best_pts = None
    best_energy = float('inf')
    
    def try_update(pts, energy):
        nonlocal best_pts, best_energy
        if pts is not None and energy < best_energy:
            best_energy = energy
            best_pts = pts
    
    # Phase 1: Fibonacci start with L-BFGS-B
    if time.time() < deadline - 10:
        pts_fib = _fibonacci_sphere(N)
        theta0, phi0 = _points_to_spherical(pts_fib)
        t0 = time.time()
        time_left = min(40.0, deadline - t0 - 5)
        pts_opt, e_opt = _optimize_spherical(theta0, phi0, time_left, t0, maxiter=5000)
        try_update(pts_opt, e_opt)
    
    # Phase 2: Random restarts with L-BFGS-B
    if time.time() < deadline - 15:
        seeds = [42, 123, 456, 789, 1024, 2048, 3141, 4321]
        for seed in seeds:
            if time.time() > deadline - 10:
                break
            pts_r = _random_sphere(N, seed)
            theta0, phi0 = _points_to_spherical(pts_r)
            t0 = time.time()
            time_left = min(20.0, deadline - t0 - 5)
            pts_opt, e_opt = _optimize_spherical(theta0, phi0, time_left, t0, maxiter=3000)
            try_update(pts_opt, e_opt)
    
    # Phase 3: Polish best solution with direct gradient descent
    if best_pts is not None and time.time() < deadline - 5:
        t0 = time.time()
        time_left = min(20.0, deadline - t0 - 2)
        pts_polished, e_polished = _refine_direct(best_pts, time_left, t0, maxiter=5000)
        try_update(pts_polished, e_polished)
    
    # Phase 4: Final L-BFGS-B polish if time remains
    if best_pts is not None and time.time() < deadline - 5:
        theta0, phi0 = _points_to_spherical(best_pts)
        t0 = time.time()
        time_left = min(15.0, deadline - t0 - 2)
        pts_opt, e_opt = _optimize_spherical(theta0, phi0, time_left, t0, maxiter=2000)
        try_update(pts_opt, e_opt)
    
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    return _project_to_sphere(best_pts)
