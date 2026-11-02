"""Optimized logarithmic energy minimizer for 100 points on the unit sphere."""
import numpy as np
import time
from math import fsum

N = 100
TIME_LIMIT = 115.0  # safety margin before 120s kill

def _pairwise_dists(X):
    """Compute pairwise chordal distances for points on unit sphere."""
    # ||x_i - x_j||^2 = 2 - 2 x_i . x_j
    G = X @ X.T
    D2 = 2.0 - 2.0 * G
    np.fill_diagonal(D2, 0.0)
    np.maximum(D2, 1e-30, out=D2)
    D = np.sqrt(D2)
    return D

def _log_energy(X):
    """Compute log(1/d) energy = -sum_{i<j} log(d_ij)."""
    D = _pairwise_dists(X)
    iu = np.triu_indices(N, k=1)
    d = D[iu]
    # log(1/d) = -log(d)
    return -fsum(np.log(d).tolist())

def _log_energy_grad(X):
    """Gradient of energy E = sum_{i<j} log(1/|x_i-x_j|) = -sum_{i<j} log|...|.
    
    dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    D = _pairwise_dists(X)
    # 1/D^2 with diagonal zeroed
    inv_D2 = np.zeros_like(D)
    iu = np.triu_indices(N, k=1)
    inv_D2[iu] = 1.0 / (D[iu] ** 2)
    inv_D2[iu[1], iu[0]] = inv_D2[iu]
    
    # grad_i = sum_j (x_i - x_j) / D_ij^2
    # = sum_j x_i / D_ij^2 - sum_j x_j / D_ij^2
    # = x_i * sum_j inv_D2[i,j] - sum_j x_j * inv_D2[i,j]
    row_sum = inv_D2.sum(axis=1)  # shape (N,)
    grad = X * row_sum[:, None] - (inv_D2 @ X)
    return grad

def _project_to_sphere(X):
    """Normalize each row to unit length."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms

def _riemannian_cg(X0, max_iter=500, tol=1e-12, time_limit=115.0):
    """Riemannian conjugate gradient on the sphere."""
    X = _project_to_sphere(X0.copy())
    E = _log_energy(X)
    
    start_time = time.time()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        grad = _log_energy_grad(X)
        # Project gradient to tangent space: g_tan = g - (g.x) x
        g_tan = grad - (grad * X).sum(axis=1, keepdims=True) * X
        
        # Check convergence
        gnorm = np.linalg.norm(g_tan)
        if gnorm < tol:
            break
        
        if it == 0:
            d = -g_tan
        else:
            # Fletcher-Reeves beta
            beta = np.dot(g_tan, g_tan) / max(prev_gnorm_sq, 1e-30)
            # Damped: cap beta
            beta = min(beta, 1.0)
            d = -g_tan + beta * d
            # Ensure d is in tangent space
            d = d - (d * X).sum(axis=1, keepdims=True) * X
        
        prev_gnorm_sq = np.dot(g_tan, g_tan)
        
        # Line search: backtracking with sufficient decrease
        alpha = 0.1
        E_new = E
        X_new = X
        best_alpha = 0.0
        best_E = E
        
        for ls in range(30):
            # Riemannian exponential map: move along d on sphere
            # exp_X(alpha * d) = X*cos(alpha*|d|) + (d/|d|)*sin(alpha*|d|)
            d_norm = np.linalg.norm(d, axis=1, keepdims=True)
            d_norm = np.maximum(d_norm, 1e-15)
            d_unit = d / d_norm
            a = alpha * d_norm
            
            cos_a = np.cos(a)
            sin_a = np.sin(a)
            
            X_trial = X * cos_a + d_unit * sin_a
            X_trial = _project_to_sphere(X_trial)
            
            # Check minimum distance
            D_trial = _pairwise_dists(X_trial)
            iu = np.triu_indices(N, k=1)
            min_d = D_trial[iu].min()
            if min_d < 1e-9:
                alpha *= 0.5
                continue
            
            E_trial = _log_energy(X_trial)
            
            if E_trial < best_E:
                best_E = E_trial
                best_alpha = alpha
                X_new = X_trial
                E_new = E_trial
            
            # Armijo condition
            if E_trial <= E + 1e-4 * alpha * np.dot(g_tan, d):
                break
            
            alpha *= 0.5
        
        if best_alpha > 0:
            X = X_new
            E = E_new
        
        if time.time() - start_time > time_limit:
            break
    
    return _project_to_sphere(X), E

def _fibonacci_spiral(offset=0.0):
    """Generate Fibonacci spiral points on sphere with offset."""
    i = np.arange(N) + 0.5 + offset
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(seed):
    """Generate random points on sphere."""
    rng = np.random.RandomState(seed)
    X = rng.randn(N, 3)
    return _project_to_sphere(X)

def construct():
    start_time = time.time()
    remaining = TIME_LIMIT - (time.time() - start_time)
    
    if remaining < 10:
        # Not enough time, return Fibonacci spiral
        return _project_to_sphere(_fibonacci_spiral())
    
    best_X = None
    best_E = float('inf')
    
    # Generate diverse initial configurations
    inits = []
    
    # Fibonacci spirals with different offsets
    for offset in [0.0, 0.1, 0.25, 0.5, 0.75]:
        inits.append(_fibonacci_spiral(offset))
    
    # Random initializations
    for seed in [42, 123, 456, 789, 1000, 2024]:
        inits.append(_random_points(seed))
    
    # Structured initializations (e.g., grid-like)
    # Two-hemisphere Fibonacci
    i1 = np.arange(N // 2) + 0.5
    z1 = 1.0 - 2.0 * i1 / (N // 2)
    phi1 = np.pi * (1.0 + np.sqrt(5.0)) * i1
    r1 = np.sqrt(np.maximum(1.0 - z1 * z1, 0.0))
    X1 = np.column_stack([r1 * np.cos(phi1), r1 * np.sin(phi1), z1])
    
    i2 = np.arange(N - N // 2) + 0.5
    z2 = -1.0 + 2.0 * i2 / (N - N // 2)
    phi2 = np.pi * (1.0 + np.sqrt(5.0)) * i2 + 0.5
    r2 = np.sqrt(np.maximum(1.0 - z2 * z2, 0.0))
    X2 = np.column_stack([r2 * np.cos(phi2), r2 * np.sin(phi2), z2])
    
    inits.append(np.vstack([X1, X2]))
    
    # Perturbed Fibonacci
    rng = np.random.RandomState(777)
    X_pert = _fibonacci_spiral() + rng.randn(N, 3) * 0.1
    inits.append(_project_to_sphere(X_pert))
    
    # Run Riemannian CG from each initialization
    per_init_time = min(remaining / len(inits), 20.0)
    
    for init_idx, X0 in enumerate(inits):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        X_opt, E_opt = _riemannian_cg(X0, max_iter=300, tol=1e-11, time_limit=TIME_LIMIT - 5)
        
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt
    
    # If we have time, do a final polish on the best solution
    if best_X is not None and time.time() - start_time < TIME_LIMIT - 10:
        X_polished, E_polished = _riemannian_cg(best_X, max_iter=200, tol=1e-13, time_limit=TIME_LIMIT - 3)
        if E_polished < best_E:
            best_X = X_polished
            best_E = E_polished
    
    if best_X is None:
        best_X = _fibonacci_spiral()
    
    return _project_to_sphere(best_X)
