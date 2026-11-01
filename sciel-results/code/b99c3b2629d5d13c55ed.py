"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _energy_and_grad(pts):
    """Compute log-energy and gradient for N points on unit sphere.
    pts: (N,3) array of unit vectors.
    Returns (energy, grad) where grad is (N,3)."""
    n = pts.shape[0]
    # Compute all pairwise distances
    # pts_i - pts_j for all i,j
    # dist^2 = 2 - 2*pts_i.dot(pts_j)
    dot = pts @ pts.T  # (N,N)
    np.fill_diagonal(dot, 1.0)
    dist_sq = 2.0 - 2.0 * dot
    np.maximum(dist_sq, 1e-30, out=dist_sq)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/dist_ij) = -sum_{i<j} log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dist[iu])
    energy = -np.sum(log_dists)
    
    # Gradient: d/dx_i [ -log|x_i - x_j| ] = (x_i - x_j) / |x_i - x_j|^2
    # grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # Note: for i<j, contribution to grad_i is (x_i - x_j)/dist^2 and to grad_j is (x_j - x_i)/dist^2
    # So grad_i = sum_{j != i} (x_i - x_j) / dist_ij^2
    
    # dist_sq is (N,N)
    # We need (x_i - x_j) / dist_sq_ij for each pair
    # diff[i,j] = x_i - x_j
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N,N,3)
    # grad[i] = sum_j (x_i - x_j) / dist_sq[i,j]
    # But dist_sq[i,i] is 0, so we mask it
    inv_dist_sq = np.zeros_like(dist_sq)
    mask = dist_sq > 1e-30
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    
    # grad[i] = sum_j diff[i,j] * inv_dist_sq[i,j]
    # Using einsum
    grad = np.einsum('ij,ijk->ik', inv_dist_sq, diff)
    
    return energy, grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _sphere_fun(x_flat):
    """Objective and gradient for scipy optimizer. x_flat is (3N,) array."""
    pts = x_flat.reshape(N, 3)
    pts = _project_to_sphere(pts)
    energy, grad = _energy_and_grad(pts)
    # Project gradient onto tangent space: grad_tan = grad - (grad . x) * x
    grad_dot_x = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tan = grad - grad_dot_x * pts
    return energy, grad_tan.flatten()


def _golden_spiral(offset=0.0):
    """Generate golden spiral points on unit sphere with optional offset."""
    i = np.arange(N) + 0.5 + offset
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _fibonacci_sphere():
    """Fibonacci sphere points."""
    return _golden_spiral(0.0)


def _random_points(rng):
    """Random points on sphere."""
    v = rng.standard_normal((N, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms


def _optimize_from_start(pts, time_limit):
    """Run L-BFGS-B optimization from starting points. Returns best energy and points."""
    start_time = time.time()
    x0 = pts.flatten()
    
    result = minimize(
        _sphere_fun,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={
            'maxiter': 5000,
            'ftol': 1e-15,
            'gtol': 1e-12,
            'maxls': 50,
            'maxcor': 20,
        }
    )
    
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return result.fun, result.x.reshape(N, 3), elapsed
    
    return result.fun, result.x.reshape(N, 3), elapsed


def _perturb_points(pts, rng, sigma=0.1):
    """Perturb points with Gaussian noise and reproject."""
    noise = rng.standard_normal(pts.shape) * sigma
    new_pts = pts + noise
    return _project_to_sphere(new_pts)


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate diverse starting configurations
    starts = []
    
    # Golden spiral
    starts.append(_golden_spiral(0.0))
    # Golden spiral with different offsets
    for offset in [0.1, 0.25, 0.5, 0.75]:
        starts.append(_golden_spiral(offset))
    
    # Random starts
    for _ in range(5):
        starts.append(_random_points(rng))
    
    # Optimize each start, keeping track of best
    best_energy = np.inf
    best_pts = starts[0]
    
    for idx, pts in enumerate(starts):
        if time.time() > deadline:
            break
        
        # Estimate time per optimization
        if idx == 0:
            time_per_opt = 20.0  # initial guess
        else:
            time_per_opt = (time.time() - start_time) / idx
        
        remaining = deadline - time.time()
        if remaining < time_per_opt * 0.5:
            break
        
        energy, opt_pts, elapsed = _optimize_from_start(pts, remaining)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = opt_pts
        
        # If we have time, try perturbation-based restarts from best
        if time.time() > deadline:
            break
        
        # Try a few perturbations from the best point found so far
        if idx >= 2:  # Do this after a few initial optimizations
            for _ in range(3):
                if time.time() > deadline:
                    break
                
                perturbed = _perturb_points(best_pts, rng, sigma=0.05)
                energy2, opt_pts2, _ = _optimize_from_start(perturbed, deadline - time.time())
                
                if energy2 < best_energy:
                    best_energy = energy2
                    best_pts = opt_pts2
    
    # Final polish: run more iterations on the best point
    if time.time() < deadline - 5.0:
        result = minimize(
            _sphere_fun,
            best_pts.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': 10000,
                'ftol': 1e-16,
                'gtol': 1e-13,
                'maxls': 100,
                'maxcor': 30,
            }
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pts = result.x.reshape(N, 3)
    
    # Ensure points are on unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
