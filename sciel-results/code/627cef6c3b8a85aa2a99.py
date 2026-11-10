"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import time
import math

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 10.0  # seconds


def _remaining_time(start_time):
    """Return remaining time before deadline."""
    return max(0.0, DEADLINE - SAFETY_MARGIN - (time.time() - start_time))


def _energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = pts.shape[0]
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Use math.fsum for accuracy
    return math.fsum(-math.log(d_i) for d_i in d)


def _energy_and_grad(pts):
    """Compute logarithmic energy and its gradient on the sphere."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    np.fill_diagonal(dist, 1.0)
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    energy = math.fsum(-math.log(d_i) for d_i in d)
    
    # Gradient: for point k, grad_k = sum_{j!=k} (x_k - x_j) / |x_k - x_j|^2
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d_sq = dist_sq[i, j]
            grad[i] += diff[i, j] / d_sq
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _tangent_space(pts, direction):
    """Project direction onto tangent space at each point."""
    # tangent = direction - (direction · p) * p
    dot = np.sum(direction * pts, axis=1, keepdims=True)
    return direction - dot * pts


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


def _adam_tangent(pts, time_limit, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam optimizer on tangent space with adaptive learning rate."""
    best_pts = pts.copy()
    best_energy = _energy(pts)
    
    n = pts.shape[0]
    m = np.zeros_like(pts)  # First moment
    v = np.zeros_like(pts)  # Second moment
    t = 0
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        t += 1
        energy, grad = _energy_and_grad(pts)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Project gradient to tangent space
        grad_tan = _tangent_space(pts, grad)
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad_tan
        v = beta2 * v + (1 - beta2) * (grad_tan * grad_tan)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Adaptive learning rate with decay
        lr_t = lr / (1 + t * 0.001)
        step = -lr_t * m_hat / (np.sqrt(v_hat) + eps)
        
        # Move in tangent direction and renormalize
        pts_new = pts + step
        pts_new = _project_to_sphere(pts_new)
        
        # Check for improvement
        energy_new = _energy(pts_new)
        if energy_new < best_energy:
            best_energy = energy_new
            best_pts = pts_new.copy()
        
        pts = pts_new
    
    return best_pts, best_energy


def _lbfgs_tangent(pts, time_limit, max_iter=2000):
    """L-BFGS optimization in Cartesian space with projection."""
    best_pts = pts.copy()
    best_energy = _energy(pts)
    start_time = time.time()
    
    def obj_and_grad(flat):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        p = flat.reshape(-1, 3)
        p = _project_to_sphere(p)
        e, g = _energy_and_grad(p)
        return e, g.flatten()
    
    try:
        result = minimize(
            fun=lambda flat: obj_and_grad(flat)[0],
            x0=pts.flatten(),
            jac=lambda flat: obj_and_grad(flat)[1],
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': 1e-10, 'ftol': 1e-12}
        )
        if not result.success and 'TimeoutError' in str(result.message):
            pass
        else:
            if result.fun < best_energy:
                best_pts = _project_to_sphere(result.x.reshape(-1, 3))
                best_energy = result.fun
    except Exception:
        pass
    
    return best_pts, best_energy


def _high_precision_polish(pts, time_limit):
    """High-precision final polish with very tight tolerances."""
    best_pts = pts.copy()
    best_energy = _energy(pts)
    start_time = time.time()
    
    def obj_and_grad(flat):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        p = flat.reshape(-1, 3)
        p = _project_to_sphere(p)
        e, g = _energy_and_grad(p)
        return e, g.flatten()
    
    try:
        result = minimize(
            fun=lambda flat: obj_and_grad(flat)[0],
            x0=pts.flatten(),
            jac=lambda flat: obj_and_grad(flat)[1],
            method='L-BFGS-B',
            options={'maxiter': 5000, 'gtol': 1e-14, 'ftol': 1e-16}
        )
        if result.fun < best_energy:
            best_pts = _project_to_sphere(result.x.reshape(-1, 3))
            best_energy = result.fun
    except Exception:
        pass
    
    return best_pts, best_energy


def _optimize_from_start(pts, time_limit):
    """Run optimization from a starting configuration with a time limit."""
    remaining = time_limit
    best_pts = pts.copy()
    best_energy = _energy(pts)
    
    # Phase 1: Adam optimizer (good for rough landscape)
    adam_time = min(remaining * 0.5, 20.0)
    if adam_time > 1.0:
        pts_adam, energy_adam = _adam_tangent(pts, adam_time)
        if energy_adam < best_energy:
            best_pts = pts_adam
            best_energy = energy_adam
        remaining = _remaining_time(time.time())
    
    # Phase 2: L-BFGS refinement
    lbfgs_time = min(remaining * 0.4, 15.0)
    if lbfgs_time > 1.0:
        pts_lbfgs, energy_lbfgs = _lbfgs_tangent(best_pts, lbfgs_time)
        if energy_lbfgs < best_energy:
            best_pts = pts_lbfgs
            best_energy = energy_lbfgs
        remaining = _remaining_time(time.time())
    
    # Phase 3: High-precision polish
    polish_time = min(remaining, 10.0)
    if polish_time > 0.5:
        pts_polish, energy_polish = _high_precision_polish(best_pts, polish_time)
        if energy_polish < best_energy:
            best_pts = pts_polish
            best_energy = energy_polish
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    
    # Generate multiple starting configurations
    starts = []
    starts.append(_fibonacci_sphere(N))
    
    # Add several random starts with different seeds
    for seed in range(8):
        if _remaining_time(start_time) < 5.0:
            break
        starts.append(_random_sphere(N, seed=seed))
    
    best_pts = starts[0].copy()
    best_energy = _energy(starts[0])
    
    # Time budget for each optimization
    num_starts = len(starts)
    time_per_opt = _remaining_time(start_time) / num_starts
    time_per_opt = min(time_per_opt, 15.0)  # Cap individual optimization time
    
    for i, start_pts in enumerate(starts):
        if _remaining_time(start_time) < 5.0:
            break
        
        pts, energy = _optimize_from_start(start_pts, time_per_opt)
        
        if energy < best_energy:
            best_pts = pts
            best_energy = energy
    
    # Final high-precision polish on the best result
    if _remaining_time(start_time) > 5.0:
        best_pts, best_energy = _high_precision_polish(best_pts, _remaining_time(start_time) - 2.0)
    
    # Final check: ensure all points are valid
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-15) or not np.all(np.isfinite(best_pts)):
        # Fallback to Fibonacci sphere
        best_pts = _fibonacci_sphere(N)
    
    # Ensure minimum distance between points
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = dist.min()
    
    if min_dist < 1e-9:
        # Perturb close points
        rng = np.random.RandomState(42)
        for _ in range(100):
            idx = np.argmin(dist)
            i, j = idx // N, idx % N
            if i == j:
                break
            # Move point j slightly
            direction = rng.randn(3)
            best_pts[j] += 1e-6 * direction
            best_pts[j] /= np.linalg.norm(best_pts[j])
            diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff * diff, axis=2))
            np.fill_diagonal(dist, np.inf)
            min_dist = dist.min()
            if min_dist >= 1e-9:
                break
    
    return best_pts
# EVOLVE-BLOCK-END
