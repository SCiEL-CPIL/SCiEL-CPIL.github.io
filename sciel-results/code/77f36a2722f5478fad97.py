"""
Optimized solution for minimizing logarithmic energy of 150 points on the sphere.
Uses Riemannian Gradient Descent with backtracking line search on the sphere manifold.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START

def _energy_and_grad(x_flat):
    """
    Computes the logarithmic energy and the Euclidean gradient.
    The gradient is then projected to the tangent space in the main loop.
    
    Energy: E = - sum_{i<j} log(d_ij)
    Gradient: grad_i = sum_{j!=i} (x_i - x_j) / d_ij^2
    """
    x = x_flat.reshape(N, 3)
    
    # Ensure points are on the sphere (defensive check)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    safe_norms = np.where(norms < 1e-12, 1.0, norms)
    x = x / safe_norms
    
    # Pairwise squared distances
    # d_ij^2 = |x_i|^2 + |x_j|^2 - 2 x_i . x_j = 2 - 2 x_i . x_j
    dot = x @ x.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0) # Avoid division by zero in gradient calc later
    dist_sq = np.maximum(dist_sq, 1e-15) # Numerical stability
    
    # Energy
    # We only need upper triangle for energy
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(dist_sq[iu])
    # Use math.fsum for higher precision summation if needed, but np.sum is usually fine for optimization
    # However, the task specifies math.fsum for the final score. For optimization, standard sum is faster.
    energy = -np.sum(np.log(d_pairs))
    
    # Gradient
    # grad_i = sum_{j!=i} (x_i - x_j) / d_ij^2
    # Let W be the matrix with W_ij = 1/d_ij^2 for i!=j, 0 for i=j
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # grad_i = x_i * sum_j W_ij - sum_j W_ij x_j
    W_sum = np.sum(W, axis=1) # sum_j W_ij
    W_x = W @ x # sum_j W_ij x_j
    
    grad_raw = x * W_sum[:, np.newaxis] - W_x
    
    return energy, grad_raw.flatten()

def _energy_only(x_flat):
    """Compute energy only."""
    e, _ = _energy_and_grad(x_flat)
    return e

def _project_to_sphere(x_flat):
    """Project points to unit sphere."""
    x = x_flat.reshape(N, 3)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    safe_norms = np.where(norms < 1e-12, 1.0, norms)
    x = x / safe_norms
    return x.flatten()

def _riemannian_gd(x_init, max_iter=2000, tol=1e-12, time_limit=10.0):
    """
    Riemannian Gradient Descent with backtracking line search.
    Update: x_new = normalize(x - alpha * grad_tangent)
    """
    start_time = time.time()
    x = x_init.copy()
    x = _project_to_sphere(x).reshape(N, 3)
    
    curr_e = _energy_only(x.flatten())
    
    # Initial step size
    alpha = 0.1 
    # Backtracking parameters
    c1 = 1e-4
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        e, grad_flat = _energy_and_grad(x.flatten())
        grad = grad_flat.reshape(N, 3)
        
        # Project gradient to tangent space
        # g_t = g - (g . x) x
        g_dot_x = np.sum(grad * x, axis=1, keepdims=True)
        grad_tangent = grad - g_dot_x * x
        
        g_norm = np.linalg.norm(grad_tangent)
        if g_norm < tol:
            break
            
        # Backtracking line search
        step = alpha
        improved = False
        for _ in range(30):
            # Riemannian update: move in direction -grad_tangent, then project back
            x_new = x - step * grad_tangent
            x_new_flat = x_new.flatten()
            x_new_flat = _project_to_sphere(x_new_flat)
            x_new = x_new_flat.reshape(N, 3)
            
            new_e = _energy_only(x_new_flat)
            
            # Armijo condition: f(x_new) <= f(x) + c1 * step * ||grad||^2
            if new_e < curr_e - c1 * step * (g_norm ** 2):
                x = x_new
                curr_e = new_e
                improved = True
                break
            step *= 0.5
            
        if not improved:
            # If no improvement, reduce step size significantly and break or continue
            alpha *= 0.5
            if alpha < 1e-10:
                break
        else:
            # If improved, we can keep alpha or increase it slightly if it was very small
            if alpha < 0.5:
                alpha *= 1.1
            elif alpha > 0.5:
                alpha = 0.5
                
    return x, curr_e

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_energy = float('inf')
    best_points = None
    
    def optimize_from(x_init, max_iter=2000, tol=1e-12, time_limit=15.0):
        """Run Riemannian GD from initial point."""
        x_opt, E_opt = _riemannian_gd(x_init, max_iter=max_iter, tol=tol, time_limit=time_limit)
        return x_opt, E_opt

    def polish(x_init, max_iter=5000, tol=1e-14, time_limit=10.0):
        """Final high-precision polish."""
        x_opt, E_opt = _riemannian_gd(x_init, max_iter=max_iter, tol=tol, time_limit=time_limit)
        return x_opt, E_opt

    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(x_fib)
    
    # 2. Perturbed Fibonacci (small noise)
    np.random.seed(42)
    noise = np.random.normal(0, 0.03, (N, 3))
    x_pert1 = x_fib + noise
    x_pert1 = x_pert1 / np.linalg.norm(x_pert1, axis=1, keepdims=True)
    initializations.append(x_pert1)
    
    # 3. Perturbed Fibonacci (medium noise)
    np.random.seed(999)
    noise = np.random.normal(0, 0.08, (N, 3))
    x_pert2 = x_fib + noise
    x_pert2 = x_pert2 / np.linalg.norm(x_pert2, axis=1, keepdims=True)
    initializations.append(x_pert2)
    
    # 4. Random points
    np.random.seed(123)
    x_rand = np.random.normal(0, 1, (N, 3))
    x_rand = x_rand / np.linalg.norm(x_rand, axis=1, keepdims=True)
    initializations.append(x_rand)
    
    # 5. Another random with different seed
    np.random.seed(777)
    x_rand2 = np.random.normal(0, 1, (N, 3))
    x_rand2 = x_rand2 / np.linalg.norm(x_rand2, axis=1, keepdims=True)
    initializations.append(x_rand2)
    
    # 6. Fibonacci with phase shift
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + 0.5
    r = np.sqrt(1.0 - z * z)
    x_fib2 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(x_fib2)
    
    # Optimize each initialization
    for x_init in initializations:
        if time.time() > deadline - 20:
            break
        
        try:
            # Coarse optimization
            x_opt, E_opt = optimize_from(x_init, max_iter=1000, tol=1e-10, time_limit=10.0)
            
            # Fine polish
            if time.time() < deadline - 5:
                x_opt, E_opt = polish(x_opt, max_iter=2000, tol=1e-14, time_limit=10.0)
                
            if E_opt < best_energy:
                best_energy = E_opt
                best_points = x_opt.copy()
        except Exception:
            continue
    
    # Final polish on the best result found so far
    if best_points is not None and time.time() < deadline - 5:
        try:
            x_opt, E_opt = polish(best_points, max_iter=5000, tol=1e-15, time_limit=15.0)
            if E_opt < best_energy:
                best_energy = E_opt
                best_points = x_opt.copy()
        except Exception:
            pass
            
    # Ensure valid output
    if best_points is None:
        # Fallback to Fibonacci
        best_points = x_fib
    
    # Final normalization
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points

# EVOLVE-BLOCK-END
