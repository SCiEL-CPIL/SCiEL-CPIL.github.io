"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for N points on unit sphere.
    Energy: sum_{i<j} log(1/|x_i - x_j|)
    Gradient: d/dx_i sum_{j!=i} log(1/|x_i - x_j|) = sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2
    """
    # pts shape (N, 3)
    # Compute pairwise distances
    # diff[i,j] = x_i - x_j
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dists = np.linalg.norm(diff, axis=2)  # (N, N)
    
    # Avoid division by zero for i==j
    np.fill_diagonal(dists, 1.0)
    
    # Energy: sum_{i<j} log(1/d_i_j) = -sum_{i<j} log(d_i_j)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    d_ij = dists[iu]
    energy = -np.sum(np.log(d_ij))
    
    # Gradient for point i: sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j, dists[i,j] = |x_i - x_j|
    # grad_i = sum_j (diff[i,j] / dists[i,j]^2)
    # But we need to be careful: the gradient of log(1/|x_i - x_j|) w.r.t. x_i is (x_i - x_j)/|x_i - x_j|^2
    # And w.r.t. x_j is (x_j - x_i)/|x_i - x_j|^2 = -(x_i - x_j)/|x_i - x_j|^2
    
    # inv_d2[i,j] = 1/|x_i - x_j|^2
    inv_d2 = 1.0 / (dists * dists)
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i] = sum_j diff[i,j] * inv_d2[i,j]
    grad = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)  # (N, 3)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _lbfgs_optimize(pts, max_iter=5000, ftol=1e-12, gtol=1e-10, maxls=20):
    """Use L-BFGS-B on the sphere via projection after each step.
    We optimize in R^3 and project back to sphere.
    """
    # Flatten for scipy
    x0 = pts.flatten()
    
    def obj_flat(x):
        p = x.reshape(N, 3)
        p = _project_to_sphere(p)
        e, g = _energy_and_grad(p)
        return e, g.flatten()
    
    # Use a wrapper that projects
    best_pts = pts.copy()
    best_energy = _energy_and_grad(pts)[0]
    
    def fun(x):
        p = x.reshape(N, 3)
        p = _project_to_sphere(p)
        e, g = _energy_and_grad(p)
        return e, g.flatten()
    
    res = minimize(fun, x0, method='L-BFGS-B', jac=True,
                   options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'maxls': maxls})
    
    if res.success or res.fun < best_energy:
        best_pts = _project_to_sphere(res.x.reshape(N, 3))
        best_energy = res.fun
    
    return best_pts, best_energy


def _fibonacci_spiral(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n, seed=42):
    """Generate n random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _energy(pts):
    """Compute just the energy (no gradient) for quick evaluation."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(N, k=1)
    d_ij = dists[iu]
    # Use math.fsum for accuracy as specified
    terms = [-math.log(d) for d in d_ij]
    return math.fsum(terms)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    # Start with Fibonacci spiral
    pts = _fibonacci_spiral(N)
    best_pts = pts.copy()
    best_energy = _energy(pts)
    
    # First optimization pass from Fibonacci spiral
    pts, e = _lbfgs_optimize(pts, max_iter=3000, ftol=1e-12, gtol=1e-10)
    if e < best_energy:
        best_pts = pts.copy()
        best_energy = e
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return _project_to_sphere(best_pts)
    
    # Try random restarts
    seeds_to_try = [1, 2, 3, 7, 13, 42, 100, 123, 456, 789]
    for seed in seeds_to_try:
        if time.time() > deadline - 15:
            break
        
        pts = _random_points(N, seed=seed)
        pts, e = _lbfgs_optimize(pts, max_iter=2000, ftol=1e-12, gtol=1e-10)
        if e < best_energy:
            best_pts = pts.copy()
            best_energy = e
    
    # Final polish with tighter tolerance on the best point
    if time.time() < deadline - 5:
        pts, e = _lbfgs_optimize(best_pts, max_iter=5000, ftol=1e-14, gtol=1e-12, maxls=30)
        if e < best_energy:
            best_pts = pts.copy()
            best_energy = e
    
    return _project_to_sphere(best_pts)

# EVOLVE-BLOCK-END
