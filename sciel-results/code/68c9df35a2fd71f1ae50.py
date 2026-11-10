"""Rewritten program: multi-start L-BFGS-B with gradient for logarithmic energy on the sphere."""
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100
DEADLINE = 145.0  # safety margin before 150s kill

def _log_energy_and_grad(points):
    """Compute logarithmic energy and its gradient for points on the unit sphere.
    
    Energy = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    where d_ij = |x_i - x_j| is the chordal distance.
    
    Gradient of -log(d_ij) w.r.t. x_i:
    d/dx_i [-log(|x_i - x_j|)] = -(x_i - x_j)/|x_i - x_j|^2
    
    But since we're constrained to the sphere, the effective gradient is the 
    tangential component. However, L-BFGS-B with the constraint handled by 
    projecting back to the sphere works well. We compute the full gradient 
    and let the optimizer handle it, then project.
    """
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    
    # Compute pairwise distances
    # diff[i,j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for i==j
    np.fill_diagonal(dist_sq, 1.0)
    
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of log(1/d_ij) = -sum over i<j of log(d_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each point i, grad_i = sum_{j!=i} (x_i - x_j)/d_ij^2
    # Because d/dx_i [-log(d_ij)] = -(x_i - x_j)/d_ij^2 ... wait let me recheck.
    # f = -log(d_ij) where d_ij = |x_i - x_j|
    # df/dx_i = -1/d_ij * d(d_ij)/dx_i = -1/d_ij * (x_i - x_j)/d_ij = -(x_i - x_j)/d_ij^2
    # So grad_i = sum_{j != i} -(x_i - x_j)/d_ij^2 = sum_{j != i} (x_j - x_i)/d_ij^2
    
    # grad[i] = sum_j (points[j] - points[i]) / dist_sq[i,j]
    # = (sum_j points[j]/dist_sq[i,j]) - points[i] * (sum_j 1/dist_sq[i,j])
    
    # Set diagonal to 0 to exclude self
    inv_dist_sq = np.zeros_like(dist_sq)
    np.fill_diagonal(dist_sq, 1.0)
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = sum_j points[j] * inv_dist_sq[i,j] - points[i] * sum_j inv_dist_sq[i,j]
    sum_j_points = points.T @ inv_dist_sq  # shape (3, n), element [k,i] = sum_j points[j][k]*inv_dist_sq[i,j]
    sum_j_inv = inv_dist_sq.sum(axis=1)  # shape (n,)
    
    grad = sum_j_points.T - points * sum_j_inv[:, np.newaxis]
    
    return energy, grad

def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return points / norms

def _fibonacci_spiral(n):
    """Generate n points on the unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, rng):
    """Generate n random points on the unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _project_to_sphere(pts)

def _perturbed_fibonacci(n, rng, scale=0.1):
    """Generate Fibonacci spiral points with random perturbation."""
    pts = _fibonacci_spiral(n).astype(np.float64)
    pts = pts + scale * rng.standard_normal((n, 3))
    return _project_to_sphere(pts)

def _optimize_from_start(points, time_budget):
    """Run L-BFGS-B optimization from a given starting point."""
    start_time = time.time()
    
    # Flatten points for L-BFGS-B
    x0 = points.ravel()
    
    def objective_and_grad(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        energy, grad = _log_energy_and_grad(pts)
        # Project gradient to tangent space
        grad = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        return energy, grad.ravel()
    
    # First evaluation
    e0, g0 = objective_and_grad(x0)
    
    # Use L-BFGS-B
    try:
        result = minimize(
            lambda x: objective_and_grad(x)[0],
            x0,
            jac=lambda x: objective_and_grad(x)[1],
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
        )
        best_points = _project_to_sphere(result.x.reshape(-1, 3))
        best_energy, _ = _log_energy_and_grad(best_points)
    except Exception:
        best_points = points
        best_energy, _ = _log_energy_and_grad(points)
    
    # Check if we should do a second polish pass
    elapsed = time.time() - start_time
    if elapsed < time_budget * 0.5:
        try:
            x1 = best_points.ravel()
            result2 = minimize(
                lambda x: objective_and_grad(x)[0],
                x1,
                jac=lambda x: objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
            )
            polished_points = _project_to_sphere(result2.x.reshape(-1, 3))
            polished_energy, _ = _log_energy_and_grad(polished_points)
            if polished_energy < best_energy:
                best_points = polished_points
                best_energy = polished_energy
        except Exception:
            pass
    
    return best_points, best_energy

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Generate multiple starting points
    starts = []
    
    # 1. Fibonacci spiral
    starts.append(_fibonacci_spiral(N))
    
    # 2. Perturbed Fibonacci (several scales)
    for scale in [0.05, 0.1, 0.2, 0.3]:
        starts.append(_perturbed_fibonacci(N, rng, scale))
    
    # 3. Random points (several)
    for _ in range(5):
        starts.append(_random_points(N, rng))
    
    # 4. More perturbed Fibonacci with different seeds
    for _ in range(5):
        starts.append(_perturbed_fibonacci(N, rng, 0.15))
    
    best_points = None
    best_energy = np.inf
    
    for idx, start_pts in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE:
            break
        
        # Allocate remaining time for this optimization
        remaining = DEADLINE - elapsed
        if remaining < 2.0:
            break
        
        # Give more time to earlier starts (they're likely better)
        if idx < 3:
            time_budget = min(remaining, 15.0)
        else:
            time_budget = min(remaining, 8.0)
        
        try:
            points, energy = _optimize_from_start(start_pts, time_budget)
            
            # Validate
            if not np.all(np.isfinite(points)):
                continue
            if points.shape != (N, 3):
                continue
            norms = np.linalg.norm(points, axis=1)
            if np.any(norms < 1e-12):
                continue
            points = _project_to_sphere(points)
            
            # Check minimum distance
            tree = cKDTree(points)
            min_dist, _ = tree.query(points, k=2)
            if np.min(min_dist[:, 1]) < 1e-9:
                continue
            
            if energy < best_energy:
                best_energy = energy
                best_points = points.copy()
        except Exception:
            continue
    
    if best_points is None:
        # Fallback to Fibonacci spiral
        best_points = _fibonacci_spiral(N)
    
    return best_points
