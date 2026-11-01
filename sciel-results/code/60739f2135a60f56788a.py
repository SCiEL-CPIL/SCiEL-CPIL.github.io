"""
Minimize logarithmic energy of 150 points on the unit sphere.
Uses a Riemannian optimization approach on the Stiefel manifold (orthonormal rows),
with multiple restarts and gradient-based local optimization.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE = 110.0  # safety margin before 120s limit

def _energy_and_grad(X):
    """
    X: (N, 3) array of points on unit sphere.
    Returns energy and gradient.
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    """
    # Compute pairwise distances
    # |x_i - x_j|^2 = 2 - 2*x_i.x_j
    G = X @ X.T  # Gram matrix
    np.fill_diagonal(G, 0.0)
    
    # Distances squared
    d2 = 2.0 - 2.0 * G
    np.fill_diagonal(d2, 0.0)
    
    # Clip to avoid log(0)
    d2 = np.maximum(d2, 1e-18)
    
    # Energy: sum over i<j of log(1/sqrt(d2_ij)) = -0.5 * sum_{i<j} log(d2_ij)
    # = -0.5 * (sum of all off-diagonal log(d2) )
    # Since d2 is symmetric, sum_{i<j} = 0.5 * sum_{i!=j}
    log_d2 = np.log(d2)
    # Sum over i<j
    iu = np.triu_indices(N, k=1)
    energy = -0.5 * np.sum(log_d2[iu])
    
    # Gradient: dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2 * (-1) * ... 
    # E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    # d/dx_i log(1/|x_i - x_j|) = d/dx_i (-log|x_i - x_j|) = -(x_i - x_j)/|x_i - x_j|^2
    # So dE/dx_i = sum_{j != i, j > i} (x_i - x_j)/|x_i - x_j|^2 + sum_{j < i} (x_j - x_i)/|x_i - x_j|^2
    #            = sum_{j != i} (x_i - x_j)/|x_i - x_j|^2
    # Wait: for pair (i,j) with i<j, contribution to E is log(1/|x_i-x_j|).
    # dE/dx_i = d/dx_i log(1/|x_i-x_j|) = -(x_i - x_j)/|x_i - x_j|^2
    # dE/dx_j = d/dx_j log(1/|x_i-x_j|) = -(x_j - x_i)/|x_i - x_j|^2 = (x_i - x_j)/|x_i - x_j|^2
    # So for each pair (i,j), grad contribution: to x_i: -(x_i-x_j)/d2_ij, to x_j: +(x_i-x_j)/d2_ij
    # Equivalently: grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2 ... let me recheck.
    
    # E = sum_{i<j} -log(|x_i - x_j|)
    # ∂E/∂x_i = sum_{j: j>i} ∂(-log|x_i-x_j|)/∂x_i + sum_{j: j<i} ∂(-log|x_j-x_i|)/∂x_i
    # ∂(-log|x_i-x_j|)/∂x_i = -(x_i - x_j)/|x_i - x_j|^2
    # ∂(-log|x_j-x_i|)/∂x_i = -(x_i - x_j)/|x_i - x_j|^2  (since |x_j-x_i|=|x_i-x_j|, derivative w.r.t x_i is -(x_i-x_j)/|...|^2)
    # So ∂E/∂x_i = sum_{j != i} -(x_i - x_j)/|x_i - x_j|^2 = sum_{j != i} (x_j - x_i)/|x_i - x_j|^2
    
    # Vectorized: grad_i = sum_j (x_j - x_i) / d2_ij for j != i
    # = (sum_j x_j / d2_ij) - x_i * sum_j (1/d2_ij)
    
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i] = sum_j (X[j] - X[i]) / d2[i,j] = sum_j X[j]/d2[i,j] - X[i] * sum_j 1/d2[i,j]
    row_sums = np.sum(inv_d2, axis=1)  # sum_j 1/d2[i,j]
    weighted_sum = inv_d2 @ X  # [i] = sum_j inv_d2[i,j] * X[j]
    
    grad = weighted_sum - np.outer(row_sums, X)
    
    return energy, grad


def _project_to_sphere(X):
    """Normalize each row to unit length."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms


def _fibonacci_spiral(n, offset=0.0):
    """Generate n points on unit sphere using golden angle spiral with offset."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n):
    """Generate n random points on unit sphere."""
    pts = np.random.randn(n, 3)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def construct():
    start_time = time.time()
    best_energy = np.inf
    best_X = None
    
    # Initial points: Fibonacci spiral
    X0 = _fibonacci_spiral(N)
    
    # Try multiple initializations and local optimizations
    inits = [X0]
    
    # Add a few random inits if time permits
    np.random.seed(42)
    for k in range(3):
        inits.append(_random_points(N))
    
    for init_idx, X_init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        X_init = _project_to_sphere(X_init)
        
        # Use L-BFGS-B with the constraint handled by re-normalization
        # We optimize in R^3 but project back to sphere after each step
        # Better: use the fact that on the sphere, we can use the Riemannian gradient
        
        # Simple approach: minimize in ambient space, project after each eval
        def obj_and_grad(X_flat):
            X = X_flat.reshape(N, 3)
            X = _project_to_sphere(X)
            e, g = _energy_and_grad(X)
            # Project gradient to tangent space: g_tan = g - (g.x) * x
            g_tan = g - np.sum(g * X, axis=1, keepdims=True) * X
            return e, g_tan.reshape(-1)
        
        x0 = X_init.reshape(-1)
        
        # Use L-BFGS-B
        try:
            res = minimize(
                obj_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'gtol': 1e-12, 'ftol': 1e-14, 'maxcor': 50}
            )
        except Exception:
            continue
        
        X_opt = res.x.reshape(N, 3)
        X_opt = _project_to_sphere(X_opt)
        
        # Final energy
        e_final, _ = _energy_and_grad(X_opt)
        
        if e_final < best_energy:
            best_energy = e_final
            best_X = X_opt
        
        # Check time
        if time.time() - start_time > DEADLINE - 30:
            break
    
    # Polish: run more iterations on the best solution found
    if best_X is not None and time.time() - start_time < DEADLINE - 10:
        X_polish = best_X.copy()
        
        def obj_and_grad2(X_flat):
            X = X_flat.reshape(N, 3)
            X = _project_to_sphere(X)
            e, g = _energy_and_grad(X)
            g_tan = g - np.sum(g * X, axis=1, keepdims=True) * X
            return e, g_tan.reshape(-1)
        
        x0 = X_polish.reshape(-1)
        try:
            res2 = minimize(
                obj_and_grad2,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 5000, 'gtol': 1e-14, 'ftol': 1e-16, 'maxcor': 100}
            )
            X_polish = res2.x.reshape(N, 3)
            X_polish = _project_to_sphere(X_polish)
            e_polish, _ = _energy_and_grad(X_polish)
            if e_polish < best_energy:
                best_energy = e_polish
                best_X = X_polish
        except Exception:
            pass
    
    if best_X is None:
        best_X = X0
    
    # Ensure valid output
    best_X = _project_to_sphere(best_X)
    
    # Check for minimum distance
    G = best_X @ best_X.T
    d2 = 2.0 - 2.0 * G
    np.fill_diagonal(d2, 0.0)
    min_d2 = np.min(d2)
    if min_d2 < (1e-9)**2:
        # Points too close, perturb
        best_X = _fibonacci_spiral(N)
        best_X = _project_to_sphere(best_X)
    
    return best_X
