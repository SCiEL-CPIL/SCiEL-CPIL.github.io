"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize as _scipy_minimize

_DEADLINE = 140.0  # safety margin before 150s kill

def _log_energy(pts):
    """Compute log(1/d) energy for all pairs using math.fsum."""
    # pts: (N,3)
    d = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(N, 1)
    return math.fsum(np.log(1.0 / dist[iu]).tolist())

def _energy_and_grad(pts):
    """Return (energy, flat gradient) for log(1/|xi-xj|) energy.
    E = sum_{i<j} -log(|xi-xj|)
    grad_i = sum_{j!=i} (xi - xj) / |xi-xj|^2   (note: positive, since we minimize -log)
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n,n,3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n,n)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # d/dxi [-log|xi-xj|] = -(xi-xj)/|xi-xj|^2
    # But we want to minimize E = sum -log(d), so grad = -sum (xi-xj)/|xi-xj|^2
    # Wait: d/dxi log(1/|xi-xj|) = d/dxi (-log|xi-xj|) = -(xi-xj)/|xi-xj|^2
    # So grad_i = -sum_{j!=i} (xi-xj)/|xi-xj|^2
    weight = -1.0 / dist_sq  # (n,n)
    np.fill_diagonal(weight, 0.0)
    grad = np.sum(weight[:, :, np.newaxis] * diff, axis=1)  # (n,3)
    
    iu = np.triu_indices(n, 1)
    energy = math.fsum(np.log(1.0 / dist[iu]).tolist())
    
    return energy, grad.flatten()

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_sphere():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _lbfgs_optimize(pts, max_time, ftol=1e-12, gtol=1e-10):
    """L-BFGS optimization using scipy with sphere projection."""
    pts = _project_to_sphere(np.array(pts, dtype=np.float64))
    x0 = pts.flatten()
    t0 = time.time()
    
    def obj(x):
        if time.time() - t0 > max_time:
            return np.inf
        p = x.reshape(N, 3)
        p = _project_to_sphere(p)
        e, _ = _energy_and_grad(p)
        return e
    
    def grad(x):
        if time.time() - t0 > max_time:
            return np.zeros_like(x)
        p = x.reshape(N, 3)
        p = _project_to_sphere(p)
        _, g = _energy_and_grad(p)
        return g
    
    try:
        res = _scipy_minimize(
            obj, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': ftol, 'gtol': gtol, 'maxfun': 20000}
        )
        pts_opt = _project_to_sphere(res.x.reshape(N, 3))
        e_opt = _log_energy(pts_opt)
        return pts_opt, e_opt
    except Exception:
        return pts, _log_energy(pts)

def _riemannian_mirror_descent(pts, max_time, lr_init=0.01):
    """Riemannian mirror descent using exponential map on the sphere.
    The exponential map on S^2: exp_p(v) = cos(|v|)*p + sin(|v|)*v/|v|
    This naturally stays on the sphere and respects curvature.
    """
    pts = _project_to_sphere(np.array(pts, dtype=np.float64))
    best_energy = _log_energy(pts)
    best_pts = pts.copy()
    lr = lr_init
    t0 = time.time()
    n = N
    iu = np.triu_indices(n, 1)
    
    def _exp_map(p, v):
        """Exponential map on S^2 at point p with tangent vector v."""
        norm_v = np.linalg.norm(v, axis=1, keepdims=True)
        # Handle zero vectors
        mask = norm_v > 1e-15
        norm_v_safe = np.where(mask, norm_v, 1.0)
        cos_term = np.cos(norm_v_safe)
        sin_term = np.sin(norm_v_safe) / norm_v_safe
        sin_term = np.where(mask, sin_term, 0.0)
        return cos_term * p + sin_term * v
    
    step = 0
    while time.time() - t0 < max_time:
        # Compute Riemannian gradient (project to tangent space)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        weight = -1.0 / dist_sq
        np.fill_diagonal(weight, 0.0)
        grad = np.sum(weight[:, :, np.newaxis] * diff, axis=1)
        
        # Project to tangent space
        grad_tan = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        # Adaptive step size with line search
        improved = False
        for attempt in range(5):
            v = -lr * grad_tan
            new_pts = _exp_map(pts, v)
            new_pts = _project_to_sphere(new_pts)  # safety projection
            
            # Quick energy check
            diff2 = new_pts[:, np.newaxis, :] - new_pts[np.newaxis, :]
            dist2 = np.sqrt(np.sum(diff2 * diff2, axis=2))
            np.fill_diagonal(dist2, 1.0)
            e2 = -np.log(dist2[iu]).sum()
            
            if e2 < best_energy - 1e-13:
                pts = new_pts
                best_energy = e2
                best_pts = pts.copy()
                lr = min(lr * 1.1, 0.5)
                improved = True
                break
            else:
                lr *= 0.5
        
        if not improved:
            if lr < 1e-12:
                break
            lr = 1e-12
        
        step += 1
    
    return best_pts, best_energy

def _polish_high_precision(pts, max_time):
    """High-precision final polish with L-BFGS at tight tolerances."""
    pts = _project_to_sphere(np.array(pts, dtype=np.float64))
    t0 = time.time()
    
    def obj(x):
        if time.time() - t0 > max_time:
            return np.inf
        p = x.reshape(N, 3)
        p = _project_to_sphere(p)
        e, _ = _energy_and_grad(p)
        return e
    
    def grad(x):
        if time.time() - t0 > max_time:
            return np.zeros_like(x)
        p = x.reshape(N, 3)
        p = _project_to_sphere(p)
        _, g = _energy_and_grad(p)
        return g
    
    x0 = pts.flatten()
    try:
        res = _scipy_minimize(
            obj, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 50000}
        )
        pts_opt = _project_to_sphere(res.x.reshape(N, 3))
        e_opt = _log_energy(pts_opt)
        return pts_opt, e_opt
    except Exception:
        return pts, _log_energy(pts)

def construct():
    t_start = time.time()
    
    # Phase 1: Fibonacci start
    pts = _fibonacci_sphere()
    best_energy = _log_energy(pts)
    best_pts = pts.copy()
    
    # Phase 2: L-BFGS optimization (main workhorse)
    remaining = _DEADLINE - time.time()
    lbfgs_time = remaining * 0.5
    if lbfgs_time > 10:
        pts, e = _lbfgs_optimize(pts, lbfgs_time, ftol=1e-10, gtol=1e-8)
        if e < best_energy:
            best_energy = e
            best_pts = pts
    
    # Phase 3: Riemannian mirror descent for refinement
    remaining = _DEADLINE - time.time()
    rmd_time = remaining * 0.3
    if rmd_time > 5:
        pts, e = _riemannian_mirror_descent(best_pts, rmd_time, lr_init=0.005)
        if e < best_energy:
            best_energy = e
            best_pts = pts
    
    # Phase 4: High-precision L-BFGS polish
    remaining = _DEADLINE - time.time()
    polish_time = remaining * 0.4
    if polish_time > 3:
        pts, e = _polish_high_precision(best_pts, polish_time)
        if e < best_energy:
            best_energy = e
            best_pts = pts
    
    # Final safety: ensure valid output
    final = _project_to_sphere(best_pts)
    # Verify no points are too close
    diff = final[:, np.newaxis, :] - final[np.newaxis, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(N, 1)
    if dist[iu].min() < 1e-9:
        # Fallback to Fibonacci if points collapsed
        final = _fibonacci_sphere()
    
    return final
# EVOLVE-BLOCK-END
