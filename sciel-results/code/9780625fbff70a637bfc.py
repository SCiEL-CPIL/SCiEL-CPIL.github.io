"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import numpy as np
import time
import math
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _energy_and_grad(X):
    """
    Compute the logarithmic energy and its gradient for points on the unit sphere.
    X: (N, 3) array of points.
    Returns: energy (float), grad (N, 3) array.
    """
    N = X.shape[0]
    # Compute pairwise distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    D2 = np.sum(diff**2, axis=2)
    
    # Avoid division by zero for i==j
    # We need D for log, so compute D but protect diagonal
    D = np.sqrt(D2)
    np.fill_diagonal(D, 1.0)
    np.fill_diagonal(D2, 1.0)
    
    # Energy: sum_{i<j} log(1/D[i,j]) = - sum_{i<j} log(D[i,j])
    # Use fsum for precision if possible, but numpy sum is usually fine for optimization.
    # The evaluator uses math.fsum on the final result.
    # For gradient descent, standard sum is sufficient.
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    log_D = np.log(D)
    energy = -np.sum(log_D[mask])
    
    # Gradient calculation
    # grad_i = sum_{j != i} (X[j] - X[i]) / D[i,j]^2
    # diff[i,j] = X[i] - X[j]
    # We want sum_j (X[j] - X[i]) / D[i,j]^2 = - sum_j (X[i] - X[j]) / D[i,j]^2
    # So grad = - np.sum(diff * (1/D2)[:, :, np.newaxis], axis=1)
    
    inv_D2 = 1.0 / D2
    np.fill_diagonal(inv_D2, 0.0)
    
    grad = -np.sum(diff * inv_D2[:, :, np.newaxis], axis=1)
    
    return energy, grad

def _project_to_sphere(X):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms

def _initial_fibonacci():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(X)

def _random_points(seed):
    """Generate random points on the sphere."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, 3))
    return _project_to_sphere(X)

def _optimize_lbfgs(X0, time_limit=10.0):
    """
    Optimize using L-BFGS-B.
    Note: L-BFGS-B works on unconstrained variables.
    We use the sphere constraint by projecting after each step?
    No, L-BFGS-B assumes a box constraint or no constraint.
    Standard approach for sphere: use the fact that the gradient on the manifold
    is the projection of the Euclidean gradient onto the tangent space.
    grad_tangent = grad - (grad . x) * x
    
    However, scipy.optimize.minimize does not natively handle manifold constraints easily
    without a custom callback or using a parameterization (like spherical coordinates).
    
    A simpler and robust approach for this specific task with N=100:
    Use the Euclidean gradient but project the points back to the sphere after each step
    in a custom gradient descent, OR use L-BFGS on the Euclidean space and hope it stays
    close, then project. But L-BFGS doesn't know about the projection.
    
    Better: Use a custom gradient descent with line search, or use L-BFGS with a
    penalty method? No, that's messy.
    
    Let's stick to a robust custom gradient descent with adaptive step size,
    or use `scipy.optimize.minimize` with `method='L-BFGS-B'` on the variables
    but manually handling the constraint is tricky.
    
    Actually, for N=100, a well-tuned gradient descent with backtracking line search
    is often very effective and simple.
    
    Let's implement a robust gradient descent with backtracking.
    """
    X = X0.copy()
    X = _project_to_sphere(X)
    energy, grad = _energy_and_grad(X)
    
    # Tangential gradient
    # grad_tan = grad - np.sum(grad * X, axis=1, keepdims=True) * X
    # This ensures we move along the surface.
    
    step_size = 0.1
    max_iter = 5000
    start_time = time.time()
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Project gradient to tangent space
        grad_tan = grad - np.sum(grad * X, axis=1, keepdims=True) * X
        
        # Backtracking line search
        alpha = step_size
        for _ in range(20):
            X_new = X - alpha * grad_tan
            X_new = _project_to_sphere(X_new)
            
            new_energy, new_grad = _energy_and_grad(X_new)
            
            if new_energy < energy:
                X = X_new
                energy = new_energy
                grad = new_grad
                # Increase step size slightly if successful
                step_size = min(alpha * 1.1, 1.0)
                break
            else:
                alpha *= 0.5
        else:
            # If line search failed, break or reduce step size significantly
            step_size *= 0.5
            if step_size < 1e-6:
                break
                
    return X, energy

def construct():
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin for evaluation overhead
    
    # Strategy: Multi-restart with L-BFGS-like optimization
    # We will try a few different initializations and pick the best.
    
    # 1. Fibonacci spiral (usually very good)
    X_fib = _initial_fibonacci()
    X_opt, E_opt = _optimize_lbfgs(X_fib, time_limit=20.0)
    
    best_X = X_opt
    best_E = E_opt
    
    # 2. Random restarts
    # Use remaining time for random restarts
    time_elapsed = time.time() - start_time
    time_remaining = time_limit - time_elapsed
    
    if time_remaining > 5.0:
        # Try a few random restarts
        # Number of restarts depends on time remaining
        # Each restart takes ~5-10s
        n_restarts = int(time_remaining / 8.0)
        n_restarts = max(1, min(n_restarts, 10)) # Cap at 10 to be safe
        
        for i in range(n_restarts):
            if time.time() - start_time > time_limit - 2.0:
                break
            
            X_rand = _random_points(seed=i + 42)
            X_opt_r, E_opt_r = _optimize_lbfgs(X_rand, time_limit=6.0)
            
            if E_opt_r < best_E:
                best_E = E_opt_r
                best_X = X_opt_r
        
        # Final polish on the best configuration found
        # Use a tighter tolerance or more iterations
        time_elapsed = time.time() - start_time
        if time.time() - start_time < time_limit - 2.0:
            # Re-optimize the best solution found with more time
            X_final, E_final = _optimize_lbfgs(best_X, time_limit=time_limit - time_elapsed - 1.0)
            if E_final < best_E:
                best_E = E_final
                best_X = X_final
    
    return best_X

# EVOLVE-BLOCK-END
