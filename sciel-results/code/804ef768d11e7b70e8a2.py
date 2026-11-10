"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100

# EVOLVE-BLOCK-START

def _fibonacci_spiral(n):
    """Generate n points on the unit sphere using a Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _halton_sphere(n, base=2):
    """Generate n points using Halton sequence on sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    pts = np.zeros((n, 3))
    for i in range(n):
        u = halton(i + 1, 2)
        v = halton(i + 1, 3)
        theta = 2.0 * np.pi * u
        cos_phi = 1.0 - 2.0 * v
        sin_phi = np.sqrt(np.maximum(0.0, 1.0 - cos_phi * cos_phi))
        pts[i, 0] = sin_phi * np.cos(theta)
        pts[i, 1] = sin_phi * np.sin(theta)
        pts[i, 2] = cos_phi
    return pts


def _log_energy_fast(pts):
    """Vectorized energy computation for speed."""
    n = len(pts)
    dots = pts @ pts.T
    sq_dists = 2.0 - 2.0 * dots
    np.fill_diagonal(sq_dists, 1.0)
    sq_dists = np.maximum(sq_dists, 1e-30)
    iu = np.triu_indices(n, 1)
    d = np.sqrt(sq_dists[iu])
    return -np.sum(np.log(d))


def _log_energy_and_grad_fast(pts):
    """Vectorized energy and gradient computation."""
    n = len(pts)
    dots = pts @ pts.T
    sq_dists = 2.0 - 2.0 * dots
    np.fill_diagonal(sq_dists, 1.0)
    sq_dists = np.maximum(sq_dists, 1e-30)
    
    d = np.sqrt(sq_dists)
    np.fill_diagonal(d, 1.0)
    
    iu = np.triu_indices(n, 1)
    energy = -np.sum(np.log(d[iu]))
    
    # grad_i = sum_{j != i} (x_i - x_j) / d_ij^2
    S = 1.0 / sq_dists
    np.fill_diagonal(S, 0.0)
    
    row_sums = S.sum(axis=1)
    grad = pts * row_sums[:, np.newaxis] - S @ pts
    
    # Vectorized tangent projection: grad[i] -= (grad[i] . pts[i]) * pts[i]
    dots_grad = np.einsum('ij,ij->i', grad, pts)
    grad -= dots_grad[:, np.newaxis] * pts
    
    return energy, grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-30)
    return pts / norms


def _optimize_from_start(x0, max_iter=300, ftol=1e-14, gtol=1e-10):
    """Run L-BFGS-B optimization from a given starting point."""
    def objective_and_grad(x_flat):
        pts = x_flat.reshape(N, 3)
        pts = _project_to_sphere(pts)
        e, g = _log_energy_and_grad_fast(pts)
        return e, g.flatten()
    
    try:
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': 20,
                'maxcor': 10
            }
        )
        pts = _project_to_sphere(result.x.reshape(N, 3))
        return pts
    except Exception:
        return None


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    # Generate diverse initializations
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(_fibonacci_spiral(N))
    
    # 2. Halton sequence
    inits.append(_halton_sphere(N))
    
    # 3-7. Random inits with different seeds
    rng = np.random.default_rng(42)
    for trial in range(5):
        pts = rng.standard_normal((N, 3))
        pts = _project_to_sphere(pts)
        inits.append(pts)
    
    # 8. Fibonacci with slight perturbation
    fib = _fibonacci_spiral(N)
    fib_perturbed = fib + 0.01 * np.random.randn(N, 3)
    inits.append(_project_to_sphere(fib_perturbed))
    
    best_pts = None
    best_energy = np.inf
    
    for init_idx, init in enumerate(inits):
        if time.time() > deadline - 40:
            break
        
        # Phase 1: Coarse optimization
        result_pts = _optimize_from_start(init.flatten(), max_iter=200, ftol=1e-12, gtol=1e-8)
        
        if result_pts is not None:
            energy = _log_energy_fast(result_pts)
            if energy < best_energy:
                best_energy = energy
                best_pts = result_pts.copy()
        
        # Phase 2: Fine optimization from best so far
        if best_pts is not None and time.time() < deadline - 15:
            result_pts2 = _optimize_from_start(best_pts.flatten(), max_iter=300, ftol=1e-14, gtol=1e-10)
            if result_pts2 is not None:
                energy2 = _log_energy_fast(result_pts2)
                if energy2 < best_energy:
                    best_energy = energy2
                    best_pts = result_pts2.copy()
    
    # Final high-precision polish
    if best_pts is not None and time.time() < deadline - 5:
        result_pts3 = _optimize_from_start(best_pts.flatten(), max_iter=500, ftol=1e-15, gtol=1e-12)
        if result_pts3 is not None:
            energy3 = _log_energy_fast(result_pts3)
            if energy3 < best_energy:
                best_energy = energy3
                best_pts = result_pts3.copy()
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    # Final check: ensure all points are valid
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-9) or not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
    
    return _project_to_sphere(best_pts)

# EVOLVE-BLOCK-END
