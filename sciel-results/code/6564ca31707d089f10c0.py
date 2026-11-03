import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
TIME_LIMIT = 110.0  # Safety margin before 120s limit

def _log_energy_and_grad(pts):
    """
    Compute logarithmic energy and its gradient for points on the unit sphere.
    Energy = sum_{i<j} log(1 / |x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    
    pts: (N, 3) array of points on the unit sphere
    Returns: (energy, gradient) where gradient is the unconstrained gradient
             (caller must project to tangent space)
    """
    n = len(pts)
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    
    # Avoid self-interaction and division by zero
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum over i<j of -log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: d/dx_i [-log(|x_i - x_j|)] = (x_i - x_j) / |x_i - x_j|^2
    # For each point i, grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # Using the diff matrix: diff[i,j] = x_i - x_j
    # grad_i = sum_j (x_i - x_j) / dist_ij^2 = sum_j diff[i,j] / dist[i,j]^2
    
    # dist^2 matrix
    dist_sq = dist * dist
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero (diagonal doesn't contribute)
    
    # Weight matrix: W[i,j] = 1 / dist[i,j]^2 for i != j, 0 for i == j
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # grad_i = sum_j W[i,j] * diff[i,j] = sum_j (1/dist_ij^2) * (x_i - x_j)
    # = (sum_j W[i,j]) * x_i - sum_j W[i,j] * x_j
    # = row_sum_i * x_i - (W @ pts)_i
    
    row_sum = W.sum(axis=1)  # (N,)
    W_pts = W @ pts  # (N, 3)
    grad = row_sum[:, np.newaxis] * pts - W_pts  # (N, 3)
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _tangent_project(pts, grad):
    """Project gradient onto tangent space at each point."""
    # tangent projection: g_t = g - (g . x) * x
    dot = np.sum(pts * grad, axis=1, keepdims=True)
    return grad - dot * pts

def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed):
    """Generate random points on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _optimize_from_start(start_pts, time_limit):
    """Run L-BFGS-B optimization from a starting configuration."""
    start_time = time.time()
    
    # Convert to flat parameterization
    x0 = start_pts.flatten()
    
    def objective(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        e, _ = _log_energy_and_grad(pts)
        return e
    
    def jac(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        _, g = _log_energy_and_grad(pts)
        # Project gradient to tangent space, then flatten
        g_t = _tangent_project(pts, g)
        return g_t.flatten()
    
    # Phase 1: Coarse optimization
    result = minimize(
        objective, x0, jac=jac,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
    )
    
    best_pts = _project_to_sphere(result.x.reshape(N, 3))
    best_energy, _ = _log_energy_and_grad(best_pts)
    
    # Phase 2: Medium precision
    elapsed = time.time() - start_time
    if elapsed < time_limit - 10:
        result2 = minimize(
            objective, result.x, jac=jac,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
        )
        pts2 = _project_to_sphere(result2.x.reshape(N, 3))
        e2, _ = _log_energy_and_grad(pts2)
        if e2 < best_energy:
            best_pts = pts2
            best_energy = e2
    
    # Phase 3: High-precision polish (final refinement)
    elapsed = time.time() - start_time
    if elapsed < time_limit - 5:
        result3 = minimize(
            objective, best_pts.flatten(), jac=jac,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 200}
        )
        pts3 = _project_to_sphere(result3.x.reshape(N, 3))
        e3, _ = _log_energy_and_grad(pts3)
        if e3 < best_energy:
            best_pts = pts3
            best_energy = e3
            
    return best_pts, best_energy

def _perturbed_fibonacci(n, scale, seed):
    """Generate Fibonacci sphere points with small random perturbations."""
    base = _fibonacci_sphere(n)
    rng = np.random.RandomState(seed)
    noise = rng.randn(n, 3) * scale
    pts = base + noise
    return _project_to_sphere(pts)

def construct():
    """Construct 150 points on the unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    
    # Best result so far
    best_pts = _fibonacci_sphere(N)
    best_energy, _ = _log_energy_and_grad(best_pts)
    
    # Generate diverse starting configurations
    starts = []
    
    # 1. Fibonacci sphere (baseline)
    starts.append(_fibonacci_sphere(N))
    
    # 2. Perturbed Fibonacci (multiple scales)
    for scale in [0.01, 0.05, 0.1, 0.2]:
        for seed in range(3):
            starts.append(_perturbed_fibonacci(N, scale, seed))
    
    # 3. Random starts
    for seed in range(5):
        starts.append(_random_sphere(N, seed))
    
    # Optimize from each start with adaptive time allocation
    num_starts = len(starts)
    for i, start in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 15:
            break
        
        # Allocate more time to later (potentially better) starts
        progress = i / max(num_starts - 1, 1)
        if progress < 0.3:
            remaining_time = min(30.0, TIME_LIMIT - elapsed - 10)
        elif progress < 0.7:
            remaining_time = min(20.0, TIME_LIMIT - elapsed - 10)
        else:
            remaining_time = min(15.0, TIME_LIMIT - elapsed - 5)
        
        if remaining_time < 5:
            break
            
        try:
            pts, energy = _optimize_from_start(start, remaining_time)
            if energy < best_energy:
                best_energy = energy
                best_pts = pts
        except Exception:
            continue
    
    # Final high-precision polish on the best configuration found
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 5:
        try:
            final_pts, final_energy = _optimize_from_start(
                best_pts, 
                TIME_LIMIT - elapsed - 2
            )
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
        except Exception:
            pass
    
    # Final projection to ensure unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    # Verify validity
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_sphere(N)
    
    # Check minimum distance
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    min_dist = np.min(dist)
    if min_dist < 1e-9:
        return _fibonacci_sphere(N)
    
    return best_pts

# EVOLVE-BLOCK-START
# (The entire logic is above; this block marker is kept for compatibility)
# EVOLVE-BLOCK-END
