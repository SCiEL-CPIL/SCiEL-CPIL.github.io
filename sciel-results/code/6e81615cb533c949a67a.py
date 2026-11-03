"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on sphere.
    pts: (N, 3) array of unit vectors.
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    """
    pts = np.asarray(pts)
    n = pts.shape[0]
    # Compute pairwise distances
    # dist^2 = 2 - 2*x_i . x_j
    dots = pts @ pts.T
    np.fill_diagonal(dots, 0.0)
    d2 = 2.0 - dots
    d2 = np.maximum(d2, 1e-30)  # avoid log(0)
    dist = np.sqrt(d2)
    
    # Energy: -sum_{i<j} log(dist_ij)
    iu = np.triu_indices(n, 1)
    log_dists = np.log(dist[iu])
    energy = -np.sum(log_dists)
    
    # Gradient: dE/dx_i = sum_{j!=i} (x_i - x_j) / dist_ij^2
    # Actually: E = -sum_{i<j} log(dist_ij)
    # dE/dx_i = -sum_{j!=i} (1/dist_ij) * (x_i - x_j)/dist_ij = -sum_{j!=i} (x_i-x_j)/dist_ij^2
    # Wait, let me recompute: d/dx_i log|xi-xj| = (xi-xj)/|xi-xj|^2
    # So dE/dx_i = -sum_{j!=i} (xi-xj)/|xi-xj|^2
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = pts[i] - pts[j]
            d = np.sqrt(np.dot(diff, diff))
            d = max(d, 1e-15)
            grad[i] += -diff / (d * d)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_spiral(n):
    """Generate fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _energy_only(pts):
    """Just compute energy for scipy minimize."""
    pts = np.asarray(pts)
    n = pts.shape[0]
    dots = pts @ pts.T
    np.fill_diagonal(dots, 0.0)
    d2 = 2.0 - dots
    d2 = np.maximum(d2, 1e-30)
    dist = np.sqrt(d2)
    iu = np.triu_indices(n, 1)
    return -np.sum(np.log(dist[iu]))


def _energy_grad_for_scipy(pts):
    """Compute energy and gradient, returning (energy, grad) for scipy."""
    energy, grad = _energy_and_grad(pts)
    return energy, grad


def _tangent_project(pts, grad):
    """Project gradient onto tangent space of sphere at each point."""
    # Tangent space at x_i: {v : v . x_i = 0}
    # Projection: v_tan = v - (v . x_i) * x_i
    dots = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - dots * pts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin
    
    # Start with fibonacci spiral
    pts = _fibonacci_spiral(N).copy()
    best_pts = pts.copy()
    best_energy = _energy_only(pts)
    
    # Phase 1: L-BFGS optimization from fibonacci start
    try:
        result = minimize(
            _energy_grad_for_scipy,
            pts.ravel(),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        if result.success or result.fun < best_energy:
            candidate = _project_to_sphere(result.x.reshape(N, 3))
            e = _energy_only(candidate)
            if e < best_energy:
                best_pts = candidate
                best_energy = e
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_pts
    
    # Phase 2: Try multiple random restarts with gradient descent + L-BFGS
    rng = np.random.default_rng(42)
    
    for restart in range(8):
        if time.time() > deadline - 20.0:
            break
            
        # Random initial configuration
        pts = rng.standard_normal((N, 3))
        pts = _project_to_sphere(pts)
        
        # Quick gradient descent to get into basin
        lr = 0.01
        for step in range(200):
            if time.time() > deadline - 10.0:
                break
            e, g = _energy_and_grad(pts)
            g = _tangent_project(pts, g)
            pts = pts - lr * g
            pts = _project_to_sphere(pts)
            lr = max(lr * 0.98, 0.001)
        
        # L-BFGS polish
        try:
            result = minimize(
                _energy_grad_for_scipy,
                pts.ravel(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            if result.success or result.fun < best_energy:
                candidate = _project_to_sphere(result.x.reshape(N, 3))
                e = _energy_only(candidate)
                if e < best_energy:
                    best_pts = candidate
                    best_energy = e
        except Exception:
            pass
    
    # Phase 3: Final ultra-precise polish on best configuration
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                _energy_grad_for_scipy,
                best_pts.ravel(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-12, 'maxls': 100}
            )
            if result.success or result.fun < best_energy:
                candidate = _project_to_sphere(result.x.reshape(N, 3))
                e = _energy_only(candidate)
                if e < best_energy:
                    best_pts = candidate
                    best_energy = e
        except Exception:
            pass
        
        # Tiny perturbation polish
        if time.time() < deadline - 2.0:
            for _ in range(5):
                if time.time() > deadline - 1.0:
                    break
                perturbed = best_pts + rng.standard_normal((N, 3)) * 1e-6
                perturbed = _project_to_sphere(perturbed)
                try:
                    result = minimize(
                        _energy_grad_for_scipy,
                        perturbed.ravel(),
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-10, 'maxls': 50}
                    )
                    if result.success or result.fun < best_energy:
                        candidate = _project_to_sphere(result.x.reshape(N, 3))
                        e = _energy_only(candidate)
                        if e < best_energy:
                            best_pts = candidate
                            best_energy = e
                except Exception:
                    pass
    
    # Final validation and normalization
    best_pts = _project_to_sphere(best_pts)
    # Ensure no points are too close
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_dist:
                min_dist = d
    
    return best_pts

# EVOLVE-BLOCK-END
