"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 5.0  # seconds


def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = pts.shape[0]
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Avoid log(0) by masking diagonal
    np.fill_diagonal(dist, 1.0)
    # Energy = sum over i<j of log(1/dist) = -sum over i<j of log(dist)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    return -np.log(dist[iu]).sum()


def _log_energy_grad(pts):
    """Compute logarithmic energy and its gradient."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    
    # Energy
    iu = np.triu_indices(n, k=1)
    energy = -np.log(dist[iu]).sum()
    
    # Gradient: d/dx_i sum_{j!=i} log(1/|x_i - x_j|) = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # For each point i, gradient contribution from pair (i,j) is (x_i - x_j)/|x_i - x_j|^2
    # We need to be careful: the full gradient for the sum over i<j of log(1/|x_i-x_j|)
    # is for point k: sum_{j!=k} (x_k - x_j) / |x_k - x_j|^2
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = dist[i, j]
            grad[i] += (pts[i] - pts[j]) / (d * d)
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points as initial configuration."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Generate random points on the unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _optimize_from_start(pts, time_limit):
    """Run optimization from a starting configuration with a time limit."""
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # Try L-BFGS-B with gradient
    def obj_and_grad(flat):
        p = flat.reshape(-1, 3)
        p = _project_to_sphere(p)
        e, g = _log_energy_grad(p)
        # The gradient needs to be projected too, but L-BFGS-B works in flat space
        # We'll just use the gradient as-is; the projection is handled by the objective
        return e, g.flatten()
    
    start_time = time.time()
    
    # Use minimize with L-BFGS-B
    try:
        result = minimize(
            fun=lambda flat: obj_and_grad(flat)[0],
            x0=pts.flatten(),
            jac=lambda flat: obj_and_grad(flat)[1],
            method='L-BFGS-B',
            options={'maxiter': 5000, 'maxfun': 5000, 'gtol': 1e-10, 'ftol': 1e-12}
        )
        if result.fun < best_energy:
            best_pts = _project_to_sphere(result.x.reshape(-1, 3))
            best_energy = result.fun
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return best_pts, best_energy
    
    # Try further refinement with tighter tolerance
    try:
        start_time2 = time.time()
        result2 = minimize(
            fun=lambda flat: obj_and_grad(flat)[0],
            x0=best_pts.flatten(),
            jac=lambda flat: obj_and_grad(flat)[1],
            method='L-BFGS-B',
            options={'maxiter': 10000, 'maxfun': 10000, 'gtol': 1e-12, 'ftol': 1e-14}
        )
        if result2.fun < best_energy:
            best_pts = _project_to_sphere(result2.x.reshape(-1, 3))
            best_energy = result2.fun
    except Exception:
        pass
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    
    # Generate multiple starting configurations
    starts = []
    starts.append(_fibonacci_sphere(N))
    
    # Add several random starts
    for seed in range(10):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        starts.append(_random_sphere(N, seed=seed))
    
    best_pts = starts[0].copy()
    best_energy = _log_energy(starts[0])
    
    time_per_opt = (DEADLINE - SAFETY_MARGIN - (time.time() - start_time)) / max(len(starts), 1)
    time_per_opt = min(time_per_opt, 15.0)  # Cap individual optimization time
    
    for i, start_pts in enumerate(starts):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        remaining = DEADLINE - SAFETY_MARGIN - (time.time() - start_time)
        t_limit = min(time_per_opt, remaining)
        
        if t_limit < 1.0:
            break
        
        pts, energy = _optimize_from_start(start_pts, t_limit)
        
        if energy < best_energy:
            best_pts = pts
            best_energy = energy
    
    # Final check: ensure all points are valid
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-15) or not np.all(np.isfinite(best_pts)):
        # Fallback to Fibonacci sphere
        best_pts = _fibonacci_sphere(N)
    
    # Ensure minimum distance between points
    # Check for points closer than 1e-9
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_dist = dist.min()
    
    if min_dist < 1e-9:
        # Perturb close points
        for _ in range(100):
            iu = np.argmin(dist)
            i, j = iu // N, iu % N
            if i == j:
                break
            # Move point j slightly
            direction = np.random.randn(3)
            best_pts[j] += 1e-6 * direction
            best_pts[j] /= np.linalg.norm(best_pts[j])
            diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            min_dist = dist.min()
            if min_dist >= 1e-9:
                break
    
    return best_pts
# EVOLVE-BLOCK-END
