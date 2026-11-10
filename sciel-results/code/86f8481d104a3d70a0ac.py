"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

_start_time = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _start_time)

def _energy_and_grad(X):
    """
    Compute logarithmic energy and gradient for points X (N, 3).
    Energy = sum_{i<j} log(1 / ||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    """
    N = X.shape[0]
    # Pairwise distances
    # dist[i,j] = ||X[i] - X[j]||
    # Use efficient vectorized computation
    X_sq = np.sum(X * X, axis=1)
    dist_sq = X_sq[:, None] + X_sq[None, :] - 2.0 * (X @ X.T)
    np.fill_diagonal(dist_sq, 0.0)
    # Clip to avoid log(0) or negative due to floating point
    dist_sq = np.maximum(dist_sq, 1e-15)
    dist = np.sqrt(dist_sq)
    
    # Mask upper triangle (i < j)
    iu = np.triu_indices(N, k=1)
    d_ij = dist[iu]
    
    # Energy: -sum log(d_ij)
    energy = -np.sum(np.log(d_ij))
    
    # Gradient:
    # E = -sum_{i<j} log(d_ij)
    # dE/dx_i = sum_{j!=i} - (1/d_ij) * (1/d_ij) * (x_i - x_j) * (-1) ? 
    # Let's derive: 
    # f = -log(||x_i - x_j||)
    # df/dx_i = - (1/||x_i - x_j||) * (1/||x_i - x_j||) * (x_i - x_j) = -(x_i - x_j) / ||x_i - x_j||^2
    # Wait: d/dx log(||x-y||) = (x-y)/||x-y||^2. So d/dx (-log(||x-y||)) = -(x-y)/||x-y||^2.
    # Gradient of E w.r.t x_i is sum_{j != i} -(x_i - x_j) / ||x_i - x_j||^2
    # Note: (x_i - x_j) / ||x_i - x_j||^2 = (x_i - x_j) / dist_sq[i,j]
    
    # Vectorized gradient:
    # G[i,j] = (x_i - x_j) / dist_sq[i,j] for i != j, 0 otherwise
    # Then grad_x_i = -sum_{j != i} G[i,j]
    
    # dist_sq is (N, N)
    # G = (X[:, None, :] - X[None, :, :]) / dist_sq[:, :, None]
    # But dist_sq has 0 on diagonal, causing division by zero.
    # We mask it out.
    
    # Safe division
    dist_sq_safe = np.where(dist_sq > 1e-15, dist_sq, 1.0)
    diff = X[:, None, :] - X[None, :, :]
    G = diff / dist_sq_safe[:, :, None]
    # Zero out diagonal
    np.fill_diagonal(G, 0.0)
    
    # grad[i] = -sum_{j != i} G[i, j]
    grad = -np.sum(G, axis=1)
    
    return energy, grad

def _retract(X, delta):
    """
    Retraction on the Stiefel manifold (orthogonal rows).
    Simple retraction: X_new = X + delta, then orthogonalize rows.
    Since we are on sphere, we can just normalize rows.
    But to be a proper Riemannian method, we use the projection to tangent space.
    Here, we use the simple retraction: normalize each row to unit length.
    """
    norms = np.linalg.norm(X + delta, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return (X + delta) / norms

def _project_tangent(X, V):
    """
    Project V onto the tangent space of the Stiefel manifold at X.
    Tangent space: {Z | X Z^T = 0} for rows? No, for rows: X is (N,3), rows are orthonormal?
    Wait, the problem is N points on S^2. They are NOT required to be orthogonal to each other.
    The manifold is (S^2)^N, the product of spheres.
    So the tangent space at X is {Z | X_i dot Z_i = 0 for all i}.
    Projection: Z_tan = Z - X * (X dot Z)
    """
    dot = np.sum(X * V, axis=1, keepdims=True)
    return V - X * dot

def _optimize_riemannian(X0, max_iter=5000, tol=1e-10):
    """
    Riemannian Conjugate Gradient / Gradient Descent with line search.
    """
    X = X0.copy()
    energy, grad = _energy_and_grad(X)
    
    # Initial search direction
    d = -_project_tangent(X, grad)
    
    for k in range(max_iter):
        if _time_left() < 5.0:
            break
        
        # Line search (backtracking)
        alpha = 1.0
        cur_energy = energy
        cur_grad = grad
        
        for ls_iter in range(30):
            X_new = _retract(X, alpha * d)
            new_energy, new_grad = _energy_and_grad(X_new)
            
            # Check for improvement
            if new_energy < cur_energy - 1e-4 * alpha * np.sum(grad * d):
                break
            alpha *= 0.5
        else:
            # If line search fails, restart
            break
            
        # Check convergence
        if np.linalg.norm(new_grad) < tol:
            X = X_new
            energy = new_energy
            grad = new_grad
            break
            
        # Update
        X = X_new
        energy = new_energy
        grad = new_grad
        
        # Project gradient to tangent space
        grad_tan = _project_tangent(X, grad)
        
        # Conjugate gradient update (Polak-Ribiere)
        if k > 0:
            # beta = <grad_new, grad_new - grad_old> / <grad_old, grad_old>
            # We need to store previous grad_tan
            # For simplicity, let's use steepest descent first to ensure stability,
            # or implement proper CG. Let's use Steepest Descent for robustness in this short budget.
            # Actually, CG is much faster. Let's implement simple CG.
            pass # We will just use steepest descent for now to avoid bugs in CG restarts
        
        # Steepest descent direction
        d = -grad_tan
        
    return X, energy

def _fibonacci_sphere():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, 3))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def construct():
    best_X = None
    best_E = float('inf')
    
    # Multi-start
    starts = []
    # 1. Fibonacci
    starts.append(_fibonacci_sphere())
    # 2. Random starts
    for s in [1, 2, 3, 4, 5]:
        starts.append(_random_sphere(s))
    
    for X0 in starts:
        if _time_left() < 10.0:
            break
            
        X_opt, E_opt = _optimize_riemannian(X0, max_iter=2000)
        
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt
            
    # Final Newton-like refinement using local quadratic approximation?
    # Or just a few more steps of steepest descent with small step size.
    if best_X is not None and _time_left() > 5.0:
        # Fine-tuning
        for _ in range(100):
            if _time_left() < 2.0:
                break
            E, G = _energy_and_grad(best_X)
            G_tan = _project_tangent(best_X, G)
            # Small step
            alpha = 0.01
            best_X = _retract(best_X, -alpha * G_tan)
            
    # Ensure unit sphere
    best_X /= np.linalg.norm(best_X, axis=1, keepdims=True)
    
    return best_X
# EVOLVE-BLOCK-END
