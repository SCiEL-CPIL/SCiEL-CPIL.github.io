"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
def _energy_and_grad(points):
    """Compute logarithmic energy and its gradient.
    Energy = -sum_{i<j} log(d_ij)
    dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    dots = points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 0.0)
    iu = np.triu_indices(N, 1)
    d2_u = d2[iu]
    d2_u = np.maximum(d2_u, 1e-30)
    d_u = np.sqrt(d2_u)
    terms = -np.log(d_u)
    energy = math.fsum(terms.tolist())
    
    # Gradient computation
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    d2_full = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2_full, 1.0)
    factor = 1.0 / d2_full
    np.fill_diagonal(factor, 0.0)
    grad = np.einsum('ij,ijk->ik', factor, diff)
    return energy, grad


def _sphere_energy_and_grad(flat):
    """Energy and gradient for scipy minimize (flat vector input)."""
    pts = flat.reshape(N, 3)
    pts = _project_to_sphere(pts)
    energy, grad = _energy_and_grad(pts)
    # Project gradient to tangent space
    g_tan = grad - (grad * pts).sum(axis=1, keepdims=True) * pts
    return energy, g_tan.flatten()


def _project_to_sphere(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _retract(pts, v):
    """Retraction on the sphere: move in direction v, then normalize."""
    new_pts = pts + v
    return _project_to_sphere(new_pts)


def _lbfgs_sphere(start_points, time_limit):
    """L-BFGS optimization on the sphere using scipy."""
    x0 = start_points.flatten().copy()
    t0 = time.time()
    
    # Use scipy L-BFGS-B with our custom objective
    result = minimize(
        _sphere_energy_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={
            'maxiter': 2000,
            'ftol': 1e-15,
            'gtol': 1e-12,
            'maxls': 40,
            'maxfun': 5000
        }
    )
    
    elapsed = time.time() - t0
    if elapsed < time_limit:
        # If we have time left, do a second pass with tighter tolerance
        x_best = result.x
        result2 = minimize(
            _sphere_energy_and_grad,
            x_best,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 1000,
                'ftol': 1e-18,
                'gtol': 1e-14,
                'maxls': 50,
                'maxfun': 3000
            }
        )
        if result2.fun < result.fun:
            result = result2
    
    best_pts = _project_to_sphere(result.x.reshape(N, 3))
    best_energy, _ = _energy_and_grad(best_pts)
    return best_pts, best_energy


def _polish(points, time_limit):
    """High-precision polish using Newton-like steps on the sphere."""
    pts = points.copy()
    t0 = time.time()
    best_f, _ = _energy_and_grad(pts)
    
    for it in range(200):
        if time.time() - t0 > time_limit:
            break
        f, grad = _energy_and_grad(pts)
        g_tan = grad - (grad * pts).sum(axis=1, keepdims=True) * pts
        g_norm = np.linalg.norm(g_tan)
        if g_norm < 1e-14:
            break
        
        # Approximate Hessian via finite differences of gradient
        # Use a simple diagonal approximation for speed
        # Actually, let's do a line search with backtracking
        alpha = 0.1
        for _ in range(30):
            new_pts = _retract(pts, -alpha * g_tan)
            new_f, _ = _energy_and_grad(new_pts)
            if new_f < f - 1e-4 * alpha * g_norm * g_norm:
                break
            alpha *= 0.5
        
        if new_f < best_f:
            best_f = new_f
            pts = new_pts
        else:
            break
    
    return pts, best_f


def _ultra_polish(points, time_limit):
    """Ultra-precise polish with very tight tolerances."""
    pts = points.copy()
    t0 = time.time()
    best_f, _ = _energy_and_grad(pts)
    
    for it in range(500):
        if time.time() - t0 > time_limit:
            break
        f, grad = _energy_and_grad(pts)
        g_tan = grad - (grad * pts).sum(axis=1, keepdims=True) * pts
        g_norm = np.linalg.norm(g_tan)
        if g_norm < 1e-16:
            break
        
        alpha = 0.01
        for _ in range(40):
            new_pts = _retract(pts, -alpha * g_tan)
            new_f, _ = _energy_and_grad(new_pts)
            if new_f < f - 1e-6 * alpha * g_norm * g_norm:
                break
            alpha *= 0.5
        
        if new_f < best_f - 1e-18:
            best_f = new_f
            pts = new_pts
        else:
            break
    
    return pts, best_f


def _deduplicate(points, min_dist=1e-9):
    """Remove duplicate points by merging very close ones."""
    pts = points.copy()
    changed = True
    while changed:
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist:
                    # Merge: average and remove one
                    pts[j] = (pts[i] + pts[j]) / 2.0
                    pts[j] = _project_to_sphere(pts[j].reshape(1, 3))[0]
                    changed = True
                    break
            if changed:
                break
    return pts


def _fibonacci_spiral():
    """Generate Fibonacci spiral points on the sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_start(seed):
    """Generate random starting points."""
    np.random.seed(seed)
    pts = np.random.randn(N, 3)
    return _project_to_sphere(pts)


def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s safety margin
    
    # Generate diverse starting points
    starts = []
    starts.append(_fibonacci_spiral())
    starts.append(_random_start(42))
    starts.append(_random_start(123))
    starts.append(_random_start(456))
    
    # Perturbed Fibonacci
    fib = _fibonacci_spiral()
    for scale in [0.05, 0.1, 0.2]:
        np.random.seed(789 + int(scale * 100))
        perturbed = fib + np.random.randn(N, 3) * scale
        starts.append(_project_to_sphere(perturbed))
    
    best_points = starts[0]
    best_energy = _energy_and_grad(best_points)[0]
    
    # Optimize from each start
    for idx, start in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 10.0:
            break
        
        time_remaining = time_limit - elapsed - 5.0
        if time_remaining < 2.0:
            break
        
        # Allocate time: 70% for main optimization, 30% for polish
        main_time = time_remaining * 0.7
        polish_time = time_remaining * 0.2
        ultra_time = time_remaining * 0.1
        
        opt_points, opt_energy = _lbfgs_sphere(start, main_time)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_points = opt_points
        
        # Polish if we have time
        elapsed = time.time() - start_time
        if elapsed < time_limit - 5.0 and polish_time > 0.5:
            polished, polish_energy = _polish(opt_points, polish_time)
            if polish_energy < best_energy:
                best_energy = polish_energy
                best_points = polished
        
        # Ultra polish the best so far
        elapsed = time.time() - start_time
        if elapsed < time_limit - 2.0 and ultra_time > 0.5:
            ultra_pts, ultra_energy = _ultra_polish(best_points, ultra_time)
            if ultra_energy < best_energy:
                best_energy = ultra_energy
                best_points = ultra_pts
    
    # Final deduplication check
    best_points = _deduplicate(best_points)
    best_points = _project_to_sphere(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
