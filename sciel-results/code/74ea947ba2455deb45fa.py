"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 100
DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _energy_fast(pts):
    """Compute log(1/d) energy using fsum for accuracy."""
    n = len(pts)
    # Compute all pairwise distances
    # pts shape (n, 3)
    # diff[i,j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Mask diagonal
    np.fill_diagonal(dist, 1.0)
    # Compute log(1/d) for i < j
    # We need sum over i < j of log(1/d_ij)
    # Use fsum for accuracy
    # Get upper triangle indices
    iu = np.triu_indices(n, 1)
    d_upper = dist[iu]
    # Filter out any zeros or very small values (shouldn't happen after normalization)
    valid = d_upper > 1e-15
    if not np.all(valid):
        # Replace invalid with 1.0 to avoid log(0)
        d_upper = np.where(valid, d_upper, 1.0)
    log_terms = np.log(1.0 / d_upper)
    return math.fsum(log_terms.tolist())

def _energy_and_grad(pts):
    """Compute energy and gradient. Gradient is sum over pairs of d_ij/(1-d_ij^2) * (pts_i - pts_j) for i != j... 
    Actually, let's derive properly.
    
    E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    
    d_ij = ||x_i - x_j||
    
    dE/dx_i = -sum_{j != i} d/dx_i log(d_ij)
            = -sum_{j != i} (1/d_ij) * d/dx_i d_ij
            = -sum_{j != i} (1/d_ij) * (x_i - x_j)/d_ij
            = -sum_{j != i} (x_i - x_j)/d_ij^2
    
    Wait, the points are on the sphere. We are minimizing over the sphere.
    The gradient on the manifold is the projection of the Euclidean gradient onto the tangent plane.
    grad_manifold = grad_euclidean - (grad_euclidean . x_i) x_i
    
    Let's compute the Euclidean gradient first.
    grad_euclidean[i] = -sum_{j != i} (x_i - x_j)/d_ij^2
    
    Then project.
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    
    # dist_sq[i,j] = ||x_i - x_j||^2
    # We want 1/d_ij^2 for i != j
    inv_dist_sq = 1.0 / dist_sq
    
    # grad_euclidean[i] = -sum_{j != i} (x_i - x_j) / d_ij^2
    # = -sum_{j != i} diff[i,j] * inv_dist_sq[i,j]
    
    # diff[i,j] is x_i - x_j
    # We want to sum over j for each i
    # grad_euclidean[i] = -sum_j (diff[i,j] * inv_dist_sq[i,j])
    
    # Using matrix multiplication:
    # grad_euclidean = -sum_j diff[i,j] * inv_dist_sq[i,j]
    # This is like - (diff * inv_dist_sq[:, :, None]).sum(axis=1)
    
    weighted_diff = diff * inv_dist_sq[:, :, None]
    grad_euclidean = -weighted_diff.sum(axis=1)
    
    # Project onto tangent plane: grad_manifold[i] = grad_euclidean[i] - (grad_euclidean[i] . x_i) x_i
    dot_prod = np.sum(grad_euclidean * pts, axis=1)
    grad_manifold = grad_euclidean - dot_prod[:, None] * pts
    
    # Energy
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(n, 1)
    d_upper = dist[iu]
    valid = d_upper > 1e-15
    d_upper = np.where(valid, d_upper, 1.0)
    log_terms = np.log(1.0 / d_upper)
    energy = math.fsum(log_terms.tolist())
    
    return energy, grad_manifold

def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=42):
    """Generate random points on sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1)
    pts /= norms[:, None]
    return pts

def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms[:, None]

def _optimize_from_start(pts, time_limit):
    """Run optimization from a starting configuration."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    
    # Use L-BFGS-B with gradient
    def objective(x):
        p = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _energy_and_grad(p)
        return e
    
    def gradient(x):
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _energy_and_grad(p)
        return g.flatten()
    
    x0 = pts.flatten()
    
    best_x = x0.copy()
    best_e = objective(x0)
    
    # Run L-BFGS-B
    try:
        res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                      options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-10})
        if res.success or res.fun < best_e:
            best_x = res.x
            best_e = res.fun
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > time_limit:
        return _project_to_sphere(best_x.reshape(-1, 3))
    
    # Do a few more refinement passes with tighter tolerance
    for _ in range(3):
        if time.time() - start_time > time_limit:
            break
        try:
            res = minimize(objective, best_x, jac=gradient, method='L-BFGS-B', 
                          options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-12})
            if res.fun < best_e:
                best_x = res.x
                best_e = res.fun
        except Exception:
            break
    
    return _project_to_sphere(best_x.reshape(-1, 3))

def construct():
    start_time = time.time()
    remaining = DEADLINE - start_time
    
    if remaining < 10:
        return _fibonacci_sphere(N)
    
    # Start with Fibonacci sphere
    pts0 = _fibonacci_sphere(N)
    best_pts = pts0.copy()
    best_energy = _energy_fast(pts0)
    
    # Try optimization from Fibonacci start
    time_per_run = remaining / 4  # Allow for multiple restarts
    pts_opt = _optimize_from_start(pts0, time_per_run)
    e_opt = _energy_fast(pts_opt)
    if e_opt < best_energy:
        best_energy = e_opt
        best_pts = pts_opt
    
    # Try random restarts
    rng = np.random.RandomState(123)
    n_restarts = 0
    while time.time() - start_time < DEADLINE - 5:
        # Random start
        pts_rand = _random_sphere(N, seed=rng.randint(0, 1000000))
        
        # Quick optimization
        remaining_time = DEADLINE - 5 - (time.time() - start_time)
        if remaining_time < 2:
            break
        
        pts_opt = _optimize_from_start(pts_rand, min(remaining_time / 2, 20.0))
        e_opt = _energy_fast(pts_opt)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt
        
        n_restarts += 1
        # If we've done enough restarts and are making progress, continue
        # Otherwise, break
        if n_restarts >= 10:
            break
    
    # Final polish if time permits
    if time.time() - start_time < DEADLINE - 10:
        pts_final = _optimize_from_start(best_pts, DEADLINE - 10 - (time.time() - start_time))
        e_final = _energy_fast(pts_final)
        if e_final < best_energy:
            best_pts = pts_final
            best_energy = e_final
    
    return _project_to_sphere(best_pts)

# EVOLVE-BLOCK-END
