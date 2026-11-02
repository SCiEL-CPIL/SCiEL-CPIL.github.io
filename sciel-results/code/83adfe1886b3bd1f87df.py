"""Minimal logarithmic energy of 100 points on the sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 150.0  # wall-clock limit; keep safety margin
SAFETY = 10.0     # stop at least this many seconds before deadline


def _log_energy(pts):
    """Compute sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)."""
    n = pts.shape[0]
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dists, 1.0)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    # Guard against zero distances
    d = np.maximum(d, 1e-15)
    return -np.sum(np.log(d))


def _log_energy_grad(pts):
    """Gradient of log energy w.r.t. points.
    
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = pts.shape[0]
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-15:
                continue
            grad[i] += d / (dist * dist)
    return grad


def _log_energy_grad_fast(pts):
    """Vectorized gradient of log energy."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    # grad[i] = sum_j (x_i - x_j) / dist_ij^2
    inv_dist2 = np.zeros_like(dists)
    mask = dists > 1e-15
    inv_dist2[mask] = 1.0 / (dists[mask] ** 2)
    np.fill_diagonal(inv_dist2, 0.0)
    grad = np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
    return grad


def _project(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_sphere(n, seed=0):
    """Fibonacci spiral initialization."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Random points on sphere."""
    v = rng.standard_normal((n, 3))
    return _project(v)


def _icosahedral_sphere(n, rng):
    """Start from icosahedral-like structure with jitter."""
    pts = _fibonacci_sphere(n)
    # Add small random perturbation
    pts = pts + rng.standard_normal((n, 3)) * 0.01
    return _project(pts)


def _optimize_from_start(pts, deadline_fn, safety):
    """Optimize from a starting configuration using L-BFGS-B on sphere coordinates.
    
    We use the parametrization: points are on S^2, so we optimize in R^3
    with a projection penalty, or use scipy's minimize with constraints.
    
    Here we use L-BFGS-B on the flat coordinates with a penalty for
    deviating from the unit sphere, then project at the end.
    """
    best_pts = _project(pts.copy())
    best_energy = _log_energy(best_pts)
    
    def objective_and_grad(flat):
        """Objective with sphere constraint penalty."""
        p = flat.reshape(-1, 3)
        # Sphere constraint penalty
        norms = np.linalg.norm(p, axis=1)
        penalty = 1e6 * np.sum((norms - 1.0) ** 2)
        # Also ensure points don't collapse
        n = p.shape[0]
        min_dist = np.inf
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(p[i] - p[j])
                if d < min_dist:
                    min_dist = d
        if min_dist < 1e-10:
            penalty += 1e10
        energy = _log_energy(_project(p)) + penalty
        
        # Gradient
        p_proj = _project(p)
        grad_proj = _log_energy_grad_fast(p_proj)
        # Chain rule through projection: dp_proj/dp = (I - p_proj p_proj^T) / |p|
        # For simplicity, use numerical check or just project the gradient
        # Actually, let's use a simpler approach: optimize directly on sphere
        # using the fact that gradient on sphere is (I - xx^T) * grad_euclidean
        grad_flat = np.zeros_like(flat)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        p_norm = p / norms
        # Project gradient to tangent space
        grad_tangent = grad_proj - np.sum(grad_proj * p_norm, axis=1, keepdims=True) * p_norm
        # Chain rule: dE/dflat = grad_tangent / |p| + 2*penalty_grad
        # For penalty: d/dp [1e6*(|p|-1)^2] = 1e6*2*(|p|-1)*p/|p|
        penalty_grad = 2e6 * (np.linalg.norm(p, axis=1, keepdims=True) - 1.0) * p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-15)
        grad_flat = (grad_tangent + penalty_grad).reshape(-1)
        
        return energy, grad_flat
    
    # Use L-BFGS-B
    flat = best_pts.reshape(-1)
    try:
        result = minimize(
            objective_and_grad,
            flat,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000}
        )
        if result.success or result.fun < best_energy:
            new_pts = _project(result.x.reshape(-1, 3))
            new_energy = _log_energy(new_pts)
            if new_energy < best_energy:
                best_pts = new_pts
                best_energy = new_energy
    except Exception:
        pass
    
    return best_pts, best_energy


def _optimize_sphere_native(pts, deadline_fn, safety):
    """Optimize directly on the sphere using Riemannian gradient descent / CG.
    
    Uses the fact that the gradient on S^2 is the Euclidean gradient
    projected onto the tangent space: grad_tangent = (I - xx^T) grad_euclidean.
    """
    best_pts = _project(pts.copy())
    best_energy = _log_energy(best_pts)
    
    # Riemannian conjugate gradient
    x = best_pts.copy()
    g = _log_energy_grad_fast(x)
    # Project to tangent space
    g = g - np.sum(g * x, axis=1, keepdims=True) * x
    
    p = -g.copy()
    alpha = 0.01
    step_size = 0.01
    max_iter = 2000
    
    for it in range(max_iter):
        if deadline_fn() > DEADLINE - SAFETY:
            break
        
        # Line search (backtracking)
        for ls in range(30):
            x_new = _project(x + step_size * p)
            e_new = _log_energy(x_new)
            if e_new < best_energy:
                best_pts = x_new.copy()
                best_energy = e_new
                break
            step_size *= 0.5
        else:
            step_size *= 0.5
            if step_size < 1e-10:
                break
        
        # Update gradient
        g_new = _log_energy_grad_fast(x_new)
        g_new = g_new - np.sum(g_new * x_new, axis=1, keepdims=True) * x_new
        
        # Check convergence
        g_norm = np.linalg.norm(g_new)
        if g_norm < 1e-10:
            break
        
        # Fletcher-Reeves beta
        beta = np.sum(g_new * g_new) / max(np.sum(g * g), 1e-15)
        p = -g_new + beta * p
        # Project p to tangent space
        p = p - np.sum(p * x_new, axis=1, keepdims=True) * x_new
        
        x = x_new
        g = g_new
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    
    def deadline_fn():
        return time.time() - start_time
    
    # Multiple diverse initializations
    rng = np.random.default_rng(42)
    
    initializations = [
        _fibonacci_sphere(N),
        _random_sphere(N, rng),
        _icosahedral_sphere(N, rng),
    ]
    
    # Add a few more random starts
    for _ in range(3):
        initializations.append(_random_sphere(N, rng))
    
    best_pts = None
    best_energy = np.inf
    
    for init in initializations:
        if deadline_fn() > DEADLINE - SAFETY - 20:
            break
        
        # First pass: Riemannian CG
        pts1, e1 = _optimize_sphere_native(init, deadline_fn, SAFETY)
        
        if e1 < best_energy:
            best_energy = e1
            best_pts = pts1.copy()
        
        # Second pass: L-BFGS-B polish from the best CG result
        if deadline_fn() > DEADLINE - SAFETY - 5:
            break
        pts2, e2 = _optimize_from_start(pts1, deadline_fn, SAFETY)
        
        if e2 < best_energy:
            best_energy = e2
            best_pts = pts2.copy()
    
    # Final polish with tighter tolerances if time permits
    if best_pts is not None and deadline_fn() < DEADLINE - SAFETY - 2:
        def final_obj_and_grad(flat):
            p = flat.reshape(-1, 3)
            p_proj = _project(p)
            energy = _log_energy(p_proj)
            grad_proj = _log_energy_grad_fast(p_proj)
            grad_tangent = grad_proj - np.sum(grad_proj * p_proj, axis=1, keepdims=True) * p_proj
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            grad_flat = (grad_tangent / norms).reshape(-1)
            return energy, grad_flat
        
        try:
            flat = best_pts.reshape(-1)
            result = minimize(
                final_obj_and_grad,
                flat,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 2000}
            )
            if result.x is not None:
                new_pts = _project(result.x.reshape(-1, 3))
                new_energy = _log_energy(new_pts)
                if new_energy < best_energy:
                    best_pts = new_pts
                    best_energy = new_energy
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    # Final validation and normalization
    best_pts = _project(best_pts)
    
    # Ensure no points are too close
    n = best_pts.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Nudge apart
                best_pts[j] += (best_pts[j] - best_pts[i]) * 1e-8
                best_pts = _project(best_pts)
                break
    
    return best_pts
