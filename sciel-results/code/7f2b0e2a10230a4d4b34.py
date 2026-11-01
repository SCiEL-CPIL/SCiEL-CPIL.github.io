"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# Deadline guard
_START_TIME = time.time()
_TIMEOUT = 110.0 # Safety margin for 120s limit

def _energy_and_grad(points):
    """
    Compute energy and Riemannian gradient for points on the sphere.
    Energy: sum_{i<j} log(1 / ||x_i - x_j||)
    """
    n = points.shape[0]
    # Compute pairwise distances
    # dists[i, j] = ||x_i - x_j||
    dists = cdist(points, points)
    # Avoid log(0)
    np.fill_diagonal(dists, 1.0)
    
    # Energy
    # We want sum_{i<j} log(1/d) = - sum_{i<j} log(d)
    # Using log-sum-exp trick or just summing upper triangle
    # For numerical stability and speed, we can sum the upper triangle
    # But cdist gives full matrix.
    # Let's compute log_dists
    # To avoid issues with very small distances, we rely on the optimization to keep them apart.
    # If distance < 1e-10, log(1/d) is huge positive, which is bad (we minimize).
    # Actually, if points are close, d is small, 1/d is large, log(1/d) is large positive.
    # We want to MINIMIZE energy. So we want distances to be LARGE.
    # Wait, the task says "minimize the logarithmic energy... sum log(1/distance)".
    # If d -> 0, log(1/d) -> +inf.
    # If d -> 2 (max), log(1/2) = -log(2).
    # So we want to maximize distances.
    # The energy is negative for well-separated points.
    # Example: d=1, log(1)=0. d=2, log(0.5) < 0.
    # So minimizing energy means maximizing distances.
    
    # Mask upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dists[mask]
    
    # Energy
    energy = np.sum(np.log(1.0 / d))
    
    # Gradient
    # E = - sum_{i<j} log(||x_i - x_j||)
    # dE/dx_i = - sum_{j != i} (1 / ||x_i - x_j||) * (x_i - x_j) / ||x_i - x_j||
    # dE/dx_i = - sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    # Note: This is the Euclidean gradient.
    # The Riemannian gradient is the projection of the Euclidean gradient onto the tangent space.
    # grad_R E(x) = P_x (grad_E E(x)) = grad_E E(x) - (grad_E E(x) . x) x
    
    grad_euclidean = np.zeros_like(points)
    # Vectorized computation of gradient
    # dists_sq = dists ** 2
    # We need sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    # Let's compute matrix G where G[i,j] = (x_i - x_j) / ||x_i - x_j||^2
    # dists[i,j] is ||x_i - x_j||
    # We need to be careful with i=j term (division by zero).
    # Set diagonal to 0.
    dists_safe = dists.copy()
    np.fill_diagonal(dists_safe, 1.0)
    
    # inv_dists_sq = 1 / (dists_safe ** 2)
    inv_dists_sq = 1.0 / (dists_safe ** 2)
    np.fill_diagonal(inv_dists_sq, 0.0)
    
    # grad_euclidean[i] = sum_j (x_i - x_j) * inv_dists_sq[i,j]
    # x_i is repeated n times.
    # (x_i - x_j) = x_i[:, None] - x_j[None, :]
    # This creates a (n, n, 3) array.
    # diff = points[:, None, :] - points[None, :, :]
    # grad_euclidean = np.sum(diff * inv_dists_sq[:, :, None], axis=1)
    
    diff = points[:, None, :] - points[None, :, :]
    grad_euclidean = np.sum(diff * inv_dists_sq[:, :, None], axis=1)
    
    # Project to tangent space
    # proj = grad_euclidean - np.sum(grad_euclidean * points, axis=1)[:, None] * points
    grad_riemannian = grad_euclidean - np.sum(grad_euclidean * points, axis=1)[:, None] * points
    
    return energy, grad_riemannian

def _retract(points, grad, step_size):
    """
    Retraction from tangent space to sphere.
    x_new = (x - step * grad) / ||x - step * grad||
    """
    v = points - step_size * grad
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.maximum(norms, 1e-15)
    return v / norms

def _optimize_lbfgs(points):
    """
    Optimize points on the sphere using L-BFGS-B on the manifold via retraction.
    """
    # Flatten points for scipy minimize
    x0 = points.flatten()
    
    def fun_and_grad(x):
        p = x.reshape(N, 3)
        # Normalize to sphere to handle any drift
        p = p / np.linalg.norm(p, axis=1, keepdims=True)
        e, g = _energy_and_grad(p)
        return e, g.flatten()
    
    # Use L-BFGS-B
    # We don't have bounds in the standard sense for sphere, but we can use it on the flattened vector.
    # However, L-BFGS-B assumes Euclidean space.
    # A better approach for manifold optimization is to use a dedicated library or implement a simple Riemannian gradient descent with line search.
    # Given time constraints and simplicity, let's use a simple Riemannian Gradient Descent with backtracking line search.
    # It's more robust to implement correctly for the sphere than hacking L-BFGS-B.
    
    # Let's switch to Riemannian Gradient Descent with backtracking.
    # It's simple and effective.
    
    best_p = points.copy()
    best_e, _ = _energy_and_grad(best_p)
    
    # Adaptive step size
    step = 0.01
    max_iter = 1000
    
    for it in range(max_iter):
        if time.time() - _START_TIME > _TIMEOUT:
            break
            
        e, g = _energy_and_grad(best_p)
        if e < best_e:
            best_e = e
            # Reset step size if we made progress? Or keep it.
            
        # Check convergence
        if np.linalg.norm(g) < 1e-8:
            break
            
        # Backtracking line search
        accepted = False
        for _ in range(20):
            new_p = _retract(best_p, g, step)
            new_e, _ = _energy_and_grad(new_p)
            if new_e < e:
                best_p = new_p
                best_e = new_e
                accepted = True
                break
            step *= 0.5
        
        if not accepted:
            # If we couldn't make progress, reduce step size significantly and try again
            step *= 0.1
            if step < 1e-10:
                break
    return best_p

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=42):
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points

def _icosahedral_lattice(n):
    """
    Generate points based on icosahedral symmetry.
    This is a heuristic to get good initial spreads.
    """
    # Start with icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # If n <= 12, just return subset
    if n <= 12:
        return verts[:n]
        
    # For n > 12, we can use the icosahedron as a seed and add more points randomly or via subdivision.
    # Simple approach: Start with icosahedron, then add random points and optimize.
    # Or, use a "spherical code" approach.
    # Let's just use random points for diversity in the multi-start.
    return _random_sphere(n, seed=123)

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_points = None
    best_energy = float('inf')
    
    # Generate diverse starts
    starts = []
    
    # 1. Fibonacci
    starts.append(_fibonacci_sphere(N))
    
    # 2. Random (multiple seeds)
    for seed in [42, 123, 999, 31415]:
        starts.append(_random_sphere(N, seed=seed))
        
    # 3. Icosahedral-inspired (actually just random for now, but labeled for clarity)
    # Let's add a "clustered" start to test robustness
    rng = np.random.RandomState(1)
    center = rng.randn(3)
    center /= np.linalg.norm(center)
    cluster_points = center + 0.1 * rng.randn(N, 3)
    cluster_points /= np.linalg.norm(cluster_points, axis=1, keepdims=True)
    starts.append(cluster_points)
    
    # Optimize each start
    for start in starts:
        if time.time() - _START_TIME > _TIMEOUT - 5:
            break
            
        optimized = _optimize_lbfgs(start)
        
        # Check validity
        if not np.all(np.isfinite(optimized)):
            continue
            
        # Check minimum distance
        dists = cdist(optimized, optimized)
        np.fill_diagonal(dists, 1.0)
        min_dist = np.min(dists)
        if min_dist < 1e-9:
            continue
            
        e, _ = _energy_and_grad(optimized)
        
        if e < best_energy:
            best_energy = e
            best_points = optimized
            
    if best_points is None:
        # Fallback to Fibonacci
        best_points = _fibonacci_sphere(N)
        
    # Final normalization
    best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
