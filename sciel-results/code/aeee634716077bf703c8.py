"""Optimize 150 points on unit sphere to minimize logarithmic energy."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE = 145.0  # safety margin before 150s kill


def _energy_and_grad(X):
    """Compute log energy and gradient for points on unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. each point (projected to tangent space).
    """
    # X shape (N, 3)
    # Pairwise distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    
    # Avoid log(0)
    dist = np.maximum(dist, 1e-15)
    
    # Energy: sum over i<j of log(1/d) = -sum_{i<j} log(d)
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    energy = -np.sum(np.log(d))
    
    # Gradient: d/dx_i [ -sum_{j != i} log(|x_i - x_j|) ]
    # = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    # For the full gradient on R^3, then project to tangent space.
    
    # grad_i = sum_{j != i} (x_j - x_i) / dist_ij^2
    # Using matrix form:
    # inv_dist2 = 1/dist^2 with diagonal 0
    inv_dist2 = np.zeros_like(dist)
    np.fill_diagonal(inv_dist2, 0.0)
    inv_dist2 = 1.0 / (dist * dist)
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad[i] = sum_j (X[j] - X[i]) * inv_dist2[i,j]
    # = sum_j X[j] * inv_dist2[i,j] - X[i] * sum_j inv_dist2[i,j]
    row_sum = inv_dist2.sum(axis=1)  # (N,)
    grad = inv_dist2 @ X - X * row_sum[:, np.newaxis]  # (N, 3)
    
    # Project gradient to tangent space: grad_tan = grad - (grad . x) * x
    dot = (grad * X).sum(axis=1)  # (N,)
    grad_tan = grad - dot[:, np.newaxis] * X
    
    return energy, grad_tan


def _project_to_sphere(X):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms


def _fibonacci_sphere(n, rng=None):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate n random points on sphere."""
    v = rng.standard_normal((n, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms


def _gradient_descent(X, max_iters=5000, lr_init=0.05, lr_max=0.15, 
                      backtracking_attempts=10, ftol=1e-14, deadline=None):
    """Gradient descent on sphere with adaptive learning rate."""
    X = _project_to_sphere(X.copy())
    energy, grad = _energy_and_grad(X)
    best_energy = energy
    best_X = X.copy()
    
    lr = lr_init
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try step
        for bt in range(backtracking_attempts):
            step = -lr * grad
            X_new = _project_to_sphere(X + step)
            e_new, g_new = _energy_and_grad(X_new)
            
            if np.isfinite(e_new) and e_new < energy:
                X = X_new
                energy = e_new
                grad = g_new
                if energy < best_energy:
                    best_energy = energy
                    best_X = X.copy()
                break
            else:
                lr *= 0.5
        
        # Adapt learning rate
        if bt < backtracking_attempts - 1:
            # Step was too large, keep reduced lr
            lr = max(lr, lr_init * 0.5)
        else:
            # All backtracking failed, reduce more
            lr *= 0.8
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10 or abs(energy - best_energy) < ftol * max(1.0, abs(best_energy)):
            break
        
        # Occasionally increase lr if making progress
        if it % 100 == 0 and energy < best_energy - 1e-6:
            lr = min(lr * 1.2, lr_max)
    
    return best_X, best_energy


def _lbfgs_optimize(X, deadline=None):
    """Use L-BFGS on sphere coordinates via parameterization."""
    # Use tangent space optimization: represent perturbations
    # Simpler: use scipy minimize with projected gradient
    def obj_and_grad(flat):
        X = _project_to_sphere(flat.reshape(N, 3))
        e, g = _energy_and_grad(X)
        return e, g.flatten()
    
    x0 = X.flatten()
    res = minimize(obj_and_grad, x0, method='L-BFGS-B',
                   options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-12,
                            'maxfun': 5000})
    X_opt = _project_to_sphere(res.x.reshape(N, 3))
    e_opt, _ = _energy_and_grad(X_opt)
    return X_opt, e_opt


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE
    
    rng = np.random.default_rng(42)
    
    best_X = None
    best_energy = np.inf
    
    # Multiple restarts with different initializations
    n_restarts = 16
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        if restart == 0:
            # Start with Fibonacci spiral
            X = _fibonacci_sphere(N, rng)
        elif restart == 1:
            # Slightly perturbed Fibonacci
            X = _fibonacci_sphere(N, rng)
            X = _project_to_sphere(X + rng.standard_normal((N, 3)) * 0.1)
        else:
            # Random initialization with some structure
            if restart % 2 == 0:
                X = _random_sphere(N, rng)
            else:
                # Clustered random: mix of random and Fibonacci
                X = _fibonacci_sphere(N, rng)
                X = _project_to_sphere(X + rng.standard_normal((N, 3)) * 0.5)
        
        # Phase 1: Gradient descent for coarse convergence
        X, e = _gradient_descent(X, max_iters=3000, lr_init=0.08, lr_max=0.15,
                                 backtracking_attempts=10, deadline=deadline)
        
        if e < best_energy:
            best_energy = e
            best_X = X.copy()
        
        # Phase 2: L-BFGS polish if time permits
        if time.time() < deadline - 10.0:
            X2, e2 = _lbfgs_optimize(X, deadline=deadline)
            if np.isfinite(e2) and e2 < best_energy:
                best_energy = e2
                best_X = X2.copy()
        
        # Phase 3: Final gradient polish
        if time.time() < deadline - 5.0:
            X3, e3 = _gradient_descent(X2, max_iters=5000, lr_init=0.02, lr_max=0.08,
                                       backtracking_attempts=10, ftol=1e-16,
                                       deadline=deadline)
            if np.isfinite(e3) and e3 < best_energy:
                best_energy = e3
                best_X = X3.copy()
    
    # Final check: ensure validity
    if best_X is None:
        # Fallback: Fibonacci spiral
        best_X = _fibonacci_sphere(N)
    
    # Normalize and validate
    best_X = _project_to_sphere(best_X)
    
    # Check for points too close
    diff = best_X[:, np.newaxis, :] - best_X[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    min_dist = dist[iu].min()
    
    if min_dist < 1e-9 or not np.all(np.isfinite(best_X)):
        # Fallback to Fibonacci
        best_X = _fibonacci_sphere(N)
        best_X = _project_to_sphere(best_X)
    
    return best_X
