"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

N = 150

# EVOLVE-BLOCK-START

def _energy_and_grad(x_flat):
    """Compute logarithmic energy and gradient for N points on the unit sphere.
    
    x_flat: shape (N*3,) vector of 3D coordinates.
    Returns: (energy, grad) where grad is shape (N*3,).
    """
    x = x_flat.reshape(N, 3)
    # Ensure points are on the sphere (normalize)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    x = x / norms
    
    # Compute pairwise distances
    # d_ij = ||x_i - x_j||
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    dist_ij = dist[iu]
    
    # Avoid log(0) for very small distances
    dist_safe = np.maximum(dist_ij, 1e-15)
    energy = -np.sum(np.log(dist_safe))
    
    # Gradient: d/dx_i of -sum_{j!=i} log(||x_i - x_j||)
    # = -sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    # But we need to be careful with the sphere constraint.
    # Since we normalize after each step, we compute the gradient in ambient space
    # and the optimizer handles the constraint via projection.
    
    # Actually, let's compute the gradient properly.
    # E = -sum_{i<j} log(d_ij) where d_ij = ||x_i - x_j||
    # dE/dx_i = -sum_{j != i} d/dx_i log(||x_i - x_j||)
    #         = -sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    
    grad = np.zeros_like(x)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = dist[i, j]
            if d < 1e-15:
                continue
            grad[i] += (x[i] - x[j]) / (d * d)
    
    # Note: E = -sum log(d), so dE/dx = -sum (x_i - x_j)/d^2
    # Wait, let me recompute. 
    # E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # dE/dx_i = -sum_{j != i} d/dx_i log(||x_i - x_j||)
    # d/dx_i log(||x_i - x_j||) = (x_i - x_j) / ||x_i - x_j||^2
    # So dE/dx_i = -sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    
    # But in my loop above I computed sum (x_i - x_j)/d^2, which is the negative of what I need.
    # Let me fix: grad[i] should be -sum_{j!=i} (x_i - x_j)/d^2
    
    grad = np.zeros_like(x)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = dist[i, j]
            if d < 1e-15:
                continue
            grad[i] += -(x[i] - x[j]) / (d * d)
    
    return energy, grad.flatten()


def _energy_and_grad_fast(x_flat):
    """Faster vectorized version of energy and gradient."""
    x = x_flat.reshape(N, 3)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    x = x / norms
    
    # Pairwise distances
    # ||x_i - x_j||^2 = 2 - 2*x_i·x_j
    dot = x @ x.T  # (N, N)
    dist_sq = 2.0 - 2.0 * dot
    np.maximum(dist_sq, 0.0, out=dist_sq)
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, 1.0)
    np.fill_diagonal(dist_sq, 1.0)
    
    iu = np.triu_indices(N, k=1)
    dist_ij = dist[iu]
    dist_sq_ij = dist_sq[iu]
    
    dist_safe = np.maximum(dist_ij, 1e-15)
    dist_sq_safe = np.maximum(dist_sq_ij, 1e-30)
    
    energy = -np.sum(np.log(dist_safe))
    
    # Gradient
    # dE/dx_i = -sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    # = -sum_{j != i} (x_i - x_j) / dist_sq_ij
    
    # Vectorized: for each i, sum over j != i of (x_i - x_j)/dist_sq_ij
    # grad[i] = -sum_j (x_i - x_j)/dist_sq_ij
    #          = -N*x_i/dist_sq... no, let's do it properly
    
    # Create weight matrix W where W[i,j] = 1/dist_sq_ij for i != j, 0 otherwise
    W = np.zeros((N, N))
    W[iu[0], iu[1]] = 1.0 / dist_sq_safe
    W[iu[1], iu[0]] = 1.0 / dist_sq_safe
    
    # grad[i] = -sum_j W[i,j] * (x_i - x_j)
    #          = -x_i * sum_j W[i,j] + sum_j W[i,j] * x_j
    
    row_sum = W.sum(axis=1, keepdims=True)  # (N, 1)
    Wx = W @ x  # (N, 3)
    
    grad = -x * row_sum + Wx
    
    return energy, grad.flatten()


def _project_to_sphere(x_flat):
    """Project points back to the unit sphere."""
    x = x_flat.reshape(N, 3)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (x / norms).flatten()


def _golden_spiral():
    """Generate golden spiral (Fibonacci) points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z]).flatten()


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Start with golden spiral
    x0 = _golden_spiral()
    
    # First, do a quick L-BFGS optimization
    best_x = x0.copy()
    best_e = _energy_and_grad_fast(best_x)[0]
    
    # L-BFGS with sphere projection
    def obj_and_grad(flat):
        x_proj = _project_to_sphere(flat)
        e, g = _energy_and_grad_fast(x_proj)
        return e, g
    
    res = minimize(obj_and_grad, x0, method='L-BFGS-B', 
                   options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
    if res.x is not None and np.all(np.isfinite(res.x)):
        x_candidate = _project_to_sphere(res.x)
        e_candidate = _energy_and_grad_fast(x_candidate)[0]
        if e_candidate < best_e:
            best_e = e_candidate
            best_x = x_candidate
    
    if time.time() > deadline:
        return _project_to_sphere(best_x).reshape(N, 3)
    
    # Try multiple random restarts with L-BFGS
    rng = np.random.default_rng(42)
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Generate random initial points
        x_rand = rng.standard_normal((N, 3))
        x_rand /= np.linalg.norm(x_rand, axis=1, keepdims=True)
        x_rand_flat = x_rand.flatten()
        
        res = minimize(obj_and_grad, x_rand_flat, method='L-BFGS-B',
                       options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.x is not None and np.all(np.isfinite(res.x)):
            x_candidate = _project_to_sphere(res.x)
            e_candidate = _energy_and_grad_fast(x_candidate)[0]
            if e_candidate < best_e:
                best_e = e_candidate
                best_x = x_candidate
    
    if time.time() > deadline:
        return _project_to_sphere(best_x).reshape(N, 3)
    
    # Polish with more iterations from the best solution
    res = minimize(obj_and_grad, best_x.copy(), method='L-BFGS-B',
                   options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
    if res.x is not None and np.all(np.isfinite(res.x)):
        x_candidate = _project_to_sphere(res.x)
        e_candidate = _energy_and_grad_fast(x_candidate)[0]
        if e_candidate < best_e:
            best_e = e_candidate
            best_x = x_candidate
    
    # Final validation
    result = _project_to_sphere(best_x).reshape(N, 3)
    
    # Check for valid points
    if not np.all(np.isfinite(result)):
        result = _project_to_sphere(_golden_spiral()).reshape(N, 3)
    
    # Check minimum distance
    diff = result[:, np.newaxis, :] - result[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    if min_dist < 1e-9:
        # Fallback to golden spiral
        result = _project_to_sphere(_golden_spiral()).reshape(N, 3)
    
    return result

# EVOLVE-BLOCK-END
