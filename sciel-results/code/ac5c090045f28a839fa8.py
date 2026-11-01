"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 150.0
SAFETY_MARGIN = 5.0


def _energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)."""
    # pts: (N, 3)
    # Use pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    # Avoid log(0) by masking diagonal
    mask = ~np.eye(N, dtype=bool)
    d_masked = d[mask]
    # Filter out any zero distances (shouldn't happen with valid points)
    d_masked = d_masked[d_masked > 1e-15]
    return -np.sum(np.log(d_masked))


def _energy_and_grad(pts):
    """
    Compute energy and gradient for logarithmic energy on sphere.
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_k: sum_{j != k} (x_k - x_j) / |x_k - x_j|^2
    (This is the gradient in R^3; we project to tangent space for Riemannian optimization)
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    d2 = np.sum(diff * diff, axis=2)  # (n, n)
    d = np.sqrt(np.maximum(d2, 1e-30))
    
    # Energy
    mask = ~np.eye(n, dtype=bool)
    energy = -np.sum(np.log(d[mask]))
    
    # Gradient: for each point k, grad_k = sum_{j != k} (x_k - x_j) / |x_k - x_j|^2
    # diff[i,j] = x_i - x_j, d2[i,j] = |x_i - x_j|^2
    # grad[i] = sum_j (x_i - x_j) / d2[i,j] for j != i
    inv_d2 = np.zeros_like(d2)
    inv_d2[mask] = 1.0 / d2[mask]
    # Set diagonal to 0
    np.fill_diagonal(inv_d2, 0.0)
    grad = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _riemannian_grad(pts, grad):
    """Project gradient to tangent space of sphere."""
    # Tangent space: remove component along radial direction
    radial = np.sum(pts * grad, axis=1, keepdims=True)
    return grad - radial * pts


def _lbfgs_sphere(pts, max_iter=1000, tol=1e-12):
    """
    L-BFGS-B optimization on the sphere using ambient-space approach.
    We optimize in R^3 with constraint |x_i|^2 = 1 handled by projection.
    Actually, let's use a simpler approach: optimize in ambient space and project.
    """
    def objective_and_grad(flat):
        pts = flat.reshape(N, 3)
        e, g = _energy_and_grad(pts)
        # Project gradient to tangent space
        g_tangent = _riemannian_grad(pts, g)
        return e, g_tangent.flatten()
    
    x0 = pts.flatten()
    result = minimize(
        objective_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10, 'maxfun': 5000}
    )
    pts_opt = result.x.reshape(N, 3)
    pts_opt = _project_to_sphere(pts_opt)
    return pts_opt, result


def _fibonacci_init():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_init(rng):
    """Random points on the sphere."""
    pts = rng.standard_normal((N, 3))
    return _project_to_sphere(pts)


def construct():
    start_time = time.time()
    
    # Start with Fibonacci initialization
    best_pts = _fibonacci_init()
    best_energy = _energy(best_pts)
    
    # Phase 1: L-BFGS-B polish from Fibonacci start
    try:
        pts_opt, result = _lbfgs_sphere(best_pts, max_iter=2000, tol=1e-14)
        e_opt = _energy(pts_opt)
        if e_opt < best_energy:
            best_pts = pts_opt
            best_energy = e_opt
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - SAFETY_MARGIN:
        return _project_to_sphere(best_pts)
    
    # Phase 2: Random restarts with L-BFGS-B
    rng = np.random.default_rng(42)
    n_restarts = 0
    max_restarts = 50
    
    while n_restarts < max_restarts:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        pts_rand = _random_init(rng)
        try:
            pts_opt, result = _lbfgs_sphere(pts_rand, max_iter=1000, tol=1e-12)
            e_opt = _energy(pts_opt)
            if e_opt < best_energy:
                best_pts = pts_opt
                best_energy = e_opt
        except Exception:
            pass
        
        n_restarts += 1
    
    # Phase 3: High-precision polish on best result
    if time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        try:
            pts_opt, result = _lbfgs_sphere(best_pts, max_iter=5000, tol=1e-15)
            e_opt = _energy(pts_opt)
            if e_opt < best_energy:
                best_pts = pts_opt
                best_energy = e_opt
        except Exception:
            pass
    
    # Final projection to ensure unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    # Validate: check no points are too close
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    min_dist = np.min(d)
    if min_dist < 1e-9:
        # Fall back to Fibonacci if points are too close
        best_pts = _fibonacci_init()
    
    return best_pts
