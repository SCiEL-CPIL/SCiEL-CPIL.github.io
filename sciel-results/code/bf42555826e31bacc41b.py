"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
import time


def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(points)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-9:
                return 1e10
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _log_energy_grad(points):
    """Compute gradient of logarithmic energy with respect to points on sphere (projected)."""
    n = len(points)
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(i + 1, n):
            d_vec = points[i] - points[j]
            d = np.linalg.norm(d_vec)
            if d < 1e-9:
                continue
            factor = 1.0 / (d * d)
            grad[i] -= d_vec * factor
            grad[j] += d_vec * factor
    for i in range(n):
        proj = np.dot(grad[i], points[i])
        grad[i] -= proj * points[i]
    return grad


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _fibonacci_sphere(n, seed=0):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + seed * 0.1
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _energy_flat(flat):
    pts = _normalize(flat.reshape(N, 3))
    return _log_energy(pts)


def _grad_flat(flat):
    pts = _normalize(flat.reshape(N, 3))
    g = _log_energy_grad(pts)
    return g.flatten()


def _optimize(flat, maxiter=15000, ftol=1e-15, gtol=1e-14, maxls=100):
    result = minimize(
        _energy_flat,
        flat,
        jac=_grad_flat,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': maxls}
    )
    pts = _normalize(result.x.reshape(N, 3))
    e = _log_energy(pts)
    return pts, e


def construct():
    start_time = time.time()
    time_budget = 110  # leave 10s margin
    
    best_pts = None
    best_energy = float('inf')
    
    # Generate diverse initializations
    inits = []
    
    # Fibonacci sphere
    inits.append(_fibonacci_sphere(N, 0).flatten())
    
    # Rotated Fibonacci spheres
    rng = np.random.default_rng(123)
    for k in range(5):
        base = _fibonacci_sphere(N, k)
        # Apply random rotation
        theta = rng.uniform(0, 2*np.pi)
        phi = rng.uniform(-np.pi/2, np.pi/2)
        c, s = np.cos(theta), np.sin(theta)
        R1 = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        cp, sp = np.cos(phi), np.sin(phi)
        R2 = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        R = R2 @ R1
        rotated = (base @ R.T)
        inits.append(rotated.flatten())
    
    # Perturbed Fibonacci
    base = _fibonacci_sphere(N, 0)
    for k in range(3):
        noise = rng.normal(0, 0.15, size=(N, 3))
        perturbed = _normalize(base + noise)
        inits.append(perturbed.flatten())
    
    # Random
    for k in range(2):
        rand_pts = rng.normal(0, 1, size=(N, 3))
        rand_pts = _normalize(rand_pts)
        inits.append(rand_pts.flatten())
    
    # Optimize from each initialization
    for idx, init_flat in enumerate(inits):
        if time.time() - start_time > time_budget:
            break
        try:
            pts, e = _optimize(init_flat.copy(), maxiter=10000, ftol=1e-14, gtol=1e-13)
            if e < best_energy:
                best_energy = e
                best_pts = pts.copy()
        except Exception:
            continue
    
    # Local search: perturb best and re-optimize
    if best_pts is not None:
        for k in range(5):
            if time.time() - start_time > time_budget:
                break
            noise = rng.normal(0, 0.03, size=(N, 3))
            perturbed = _normalize(best_pts + noise)
            try:
                pts, e = _optimize(perturbed.flatten(), maxiter=10000, ftol=1e-14, gtol=1e-13)
                if e < best_energy:
                    best_energy = e
                    best_pts = pts.copy()
            except Exception:
                continue
    
    # Final high-precision refinement
    if best_pts is not None and time.time() - start_time < time_budget:
        try:
            pts, e = _optimize(best_pts.flatten(), maxiter=30000, ftol=1e-16, gtol=1e-15, maxls=200)
            if e < best_energy:
                best_energy = e
                best_pts = pts.copy()
        except Exception:
            pass
    
    # Second refinement pass if time allows
    if best_pts is not None and time.time() - start_time < time_budget - 10:
        try:
            pts, e = _optimize(best_pts.flatten(), maxiter=30000, ftol=1e-16, gtol=1e-15, maxls=200)
            if e < best_energy:
                best_energy = e
                best_pts = pts.copy()
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback: return Fibonacci sphere
        best_pts = _fibonacci_sphere(N, 0)
    
    return best_pts
# EVOLVE-BLOCK-END
