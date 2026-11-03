"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100
DEADLINE = 120.0  # seconds


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    """
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Avoid division by zero
    mask = np.ones((N, N), dtype=bool)
    np.fill_diagonal(mask, False)
    dist_safe = np.where(mask, dist, 1.0)
    
    # Energy: sum over i<j of log(1/d) = -sum over i<j of log(d)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    d_ij = dist[iu]
    # Filter out very small distances
    valid = d_ij > 1e-15
    if not np.all(valid):
        d_ij = np.where(valid, d_ij, 1e-15)
    
    energy = -np.sum(np.log(d_ij))
    
    # Gradient: dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # Because d/dx_i log(1/|x_i-x_j|) = -d/dx_i log|x_i-x_j| = -(x_i-x_j)/|x_i-x_j|^2
    # Wait: E = sum_{i<j} log(1/|x_i-x_j|) = -sum_{i<j} log|x_i-x_j|
    # dE/dx_k = -sum_{j: j<k} (x_k - x_j)/|x_k-x_j|^2 - sum_{j: j>k} (x_k - x_j)/|x_k-x_j|^2
    #          = -sum_{j!=k} (x_k - x_j)/|x_k-x_j|^2
    #          = sum_{j!=k} (x_j - x_k)/|x_k-x_j|^2
    
    # Vectorized gradient computation
    # diff[i,j] = x_i - x_j, dist[i,j] = |x_i - x_j|
    # grad[i] = sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2
    #         = -sum_{j!=i} diff[i,j] / dist[i,j]^2
    
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_dist_sq = np.where(mask, 1.0 / (dist_safe ** 2), 0.0)
    
    # grad[i] = -sum_j (diff[i,j] * inv_dist_sq[i,j])
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_spiral(offset=0.0, n_rot=0.0):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(N) + 0.5 + offset
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + n_rot
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(pts)


def _random_init(seed):
    """Generate random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def construct():
    start_time = time.time()
    
    # Best result so far
    best_pts = _fibonacci_spiral()
    best_energy = _energy_and_grad(best_pts)[0]
    
    # Try multiple initial configurations
    initializations = []
    
    # Fibonacci spiral with different offsets and rotations
    for offset in [0.0, 0.25, 0.5, 0.75]:
        for n_rot in [0.0, np.pi/4, np.pi/2, 3*np.pi/4]:
            initializations.append(_fibonacci_spiral(offset, n_rot))
    
    # Random initializations
    for seed in range(5):
        initializations.append(_random_init(seed))
    
    # Optimize each initialization
    for init_idx, init_pts in enumerate(initializations):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15.0:  # Leave 15s safety margin
            break
        
        try:
            result = minimize(
                fun=lambda x: _energy_and_grad(x)[0],
                x0=init_pts.flatten(),
                jac=lambda x: _energy_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 2000, 'gtol': 1e-10, 'ftol': 1e-14, 'maxls': 50}
            )
            
            # Project back to sphere
            opt_pts = _project_to_sphere(result.x.reshape(N, 3))
            opt_energy = _energy_and_grad(opt_pts)[0]
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts
        
        except Exception:
            continue
    
    # Final polish with tighter tolerances on the best result
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 10.0:
        try:
            result = minimize(
                fun=lambda x: _energy_and_grad(x)[0],
                x0=best_pts.flatten(),
                jac=lambda x: _energy_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 5000, 'gtol': 1e-12, 'ftol': 1e-16, 'maxls': 100}
            )
            opt_pts = _project_to_sphere(result.x.reshape(N, 3))
            opt_energy = _energy_and_grad(opt_pts)[0]
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts
        except Exception:
            pass
    
    # Ensure points are on unit sphere and valid
    best_pts = _project_to_sphere(best_pts)
    
    # Validate: no points closer than 1e-9
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(N, k=1)
    min_dist = dist[iu].min()
    
    if min_dist < 1e-9:
        # Fall back to Fibonacci spiral
        best_pts = _fibonacci_spiral()
    
    # Check for non-finite values
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral()
    
    return best_pts
# EVOLVE-BLOCK-END
