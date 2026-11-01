"""Optimized logarithmic energy minimizer for 150 points on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE = 120.0
SAFETY_MARGIN = 5.0


def _energy_and_grad(x):
    """Compute logarithmic energy and gradient for points x of shape (N, 3).
    
    Energy = sum_{i<j} log(1 / ||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    """
    # x: (N, 3)
    # Compute pairwise distances
    # diff[i, j] = x[i] - x[j]
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    
    # Avoid log(0) by clipping
    dist_sq = np.maximum(dist_sq, 1e-30)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of log(1/dist) = -sum over i<j of log(dist)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    log_dists = np.log(dist[iu])
    energy = -np.sum(log_dists)
    
    # Gradient: d/dx_k of [-log(||x_i - x_j||)] for i<j
    # d/dx_k log(||x_i - x_j||) = (x_k - x_j) / ||x_i - x_j|| if k==i
    #                             = (x_k - x_i) / ||x_i - x_j|| if k==j
    # So d/dx_k [-log(||x_i - x_j||)] = -(x_k - x_j)/dist if k==i, -(x_k - x_i)/dist if k==j
    # 
    # Gradient for point k: sum over j!=k of (x_j - x_k) / ||x_k - x_j||
    # (the negative sign from -log combined with the derivative)
    
    # For each pair (i,j) with i<j:
    # grad_i gets contribution: -(x_i - x_j) / dist_ij = (x_j - x_i) / dist_ij
    # grad_j gets contribution: -(x_j - x_i) / dist_ij = (x_i - x_j) / dist_ij
    
    grad = np.zeros_like(x)
    # dist[iu] gives distances for all pairs in upper triangle
    d_ij = dist[iu]  # (N*(N-1)/2,)
    # diff[iu] gives x[i] - x[j] for each pair
    dvec = diff[iu]  # (N*(N-1)/2, 3)
    
    # Contribution to grad[i]: (x_j - x_i) / dist = -dvec / dist
    # Contribution to grad[j]: (x_i - x_j) / dist = dvec / dist
    
    # We need to scatter these contributions
    # iu[0] are the row indices (i), iu[1] are the column indices (j)
    i_idx = iu[0]
    j_idx = iu[1]
    
    contrib = dvec / d_ij[:, np.newaxis]  # (pairs, 3), this is (x_i - x_j)/dist
    
    # For point i: add -contrib (i.e., (x_j - x_i)/dist)
    # For point j: add +contrib (i.e., (x_i - x_j)/dist)
    np.add.at(grad, i_idx, -contrib)
    np.add.at(grad, j_idx, contrib)
    
    return energy, grad


def _halton_init(n, dim=3):
    """Generate n points using Halton sequence, mapped to unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    points = []
    for i in range(1, n + 1):
        p = [halton(i, b) for b in [2, 3, 5]]
        # Map to sphere using a simple method
        # Convert to angles
        theta = 2 * np.pi * p[0]
        phi = np.arccos(2 * p[1] - 1)
        pt = [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
        norm = np.linalg.norm(pt)
        points.append(pt / norm)
    return np.array(points)


def _fibonacci_init(n):
    """Generate n points using Fibonacci spiral on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _project_to_sphere(x):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return x / norms


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    # Try multiple initializations and pick the best result
    best_x = None
    best_energy = float('inf')
    
    initializations = [
        ("halton", _halton_init(N)),
        ("fibonacci", _fibonacci_init(N)),
    ]
    
    # Add a few random initializations
    np.random.seed(42)
    for _ in range(2):
        x_rand = np.random.randn(N, 3)
        x_rand = x_rand / np.linalg.norm(x_rand, axis=1, keepdims=True)
        initializations.append(("random", x_rand))
    
    for name, x0 in initializations:
        if time.time() > deadline:
            break
        
        x0 = _project_to_sphere(x0)
        
        # Run BFGS optimization
        try:
            result = minimize(
                _energy_and_grad,
                x0.flatten(),
                jac=True,
                method='BFGS',
                options={'maxiter': 2000, 'gtol': 1e-10, 'disp': False}
            )
            
            x_opt = _project_to_sphere(result.x.reshape(N, 3))
            
            # Verify and compute final energy
            energy, _ = _energy_and_grad(x_opt)
            
            # Check validity
            if not np.all(np.isfinite(x_opt)):
                continue
            if np.any(np.linalg.norm(x_opt, axis=1) < 1e-15):
                continue
            
            # Check minimum distance
            diff = x_opt[:, np.newaxis, :] - x_opt[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            np.fill_diagonal(dist_sq, np.inf)
            min_dist_sq = np.min(dist_sq)
            if min_dist_sq < (1e-9) ** 2:
                continue
            
            if energy < best_energy:
                best_energy = energy
                best_x = x_opt.copy()
                
        except Exception:
            continue
    
    if best_x is None:
        # Fallback to fibonacci
        best_x = _fibonacci_init(N)
    
    return _project_to_sphere(best_x)
