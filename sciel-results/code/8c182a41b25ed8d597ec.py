"""Optimized logarithmic energy minimization for 100 points on the sphere."""
import numpy as np
import math
import time

N = 100
_DEADLINE = 150.0  # wall-clock limit
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START) - 2.0  # 2s safety margin


def _log_energy(X):
    """Compute sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)."""
    # Pairwise distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Avoid log(0) on diagonal
    np.fill_diagonal(dist, 1.0)
    # Sum over upper triangle
    iu = np.triu_indices(N, 1)
    vals = -np.log(dist[iu])
    return math.fsum(vals.tolist())


def _log_energy_grad(X):
    """Compute gradient of log energy w.r.t. X (on R^3, not projected)."""
    # diff[i,j] = X[i] - X[j]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    
    # Gradient: d/dX_i sum_{j != i} log(1/|X_i - X_j|) = -sum_{j != i} (X_i - X_j)/|X_i - X_j|^2
    # = sum_{j != i} (X_j - X_i)/|X_i - X_j|^2
    # inv_dist_sq[i,j] = 1/|X_i - X_j|^2
    inv_dist_sq = 1.0 / (dist * dist)
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = sum_j (X_j - X_i) / |X_i - X_j|^2
    # = sum_j X_j * inv_dist_sq[i,j] - X_i * sum_j inv_dist_sq[i,j]
    row_sum = inv_dist_sq.sum(axis=1, keepdims=True)
    grad = X.T @ inv_dist_sq - X * row_sum
    return grad


def _project_to_sphere(X):
    """Normalize each row to unit length."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms


def _golden_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _sobol_sphere(n, seed=0):
    """Generate points on sphere using quasi-random directions."""
    rng = np.random.default_rng(seed)
    # Use random normal vectors normalized
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _optimize_from_start(X0, max_iter=2000, tol=1e-12):
    """Gradient descent with backtracking line search on the sphere."""
    X = X0.copy()
    X = _project_to_sphere(X)
    best_energy = _log_energy(X)
    best_X = X.copy()
    
    step = 0.01
    for it in range(max_iter):
        if _remaining() < 5.0:
            break
        
        grad = _log_energy_grad(X)
        # Project gradient to tangent space: grad - (grad . x) x
        proj_grad = grad - (grad * X).sum(axis=1, keepdims=True) * X
        
        # Check convergence
        gnorm = np.linalg.norm(proj_grad)
        if gnorm < tol:
            break
        
        # Backtracking line search
        alpha = step
        cur_energy = best_energy
        improved = False
        for ls in range(30):
            # Move in tangent direction
            X_trial = X + alpha * proj_grad
            X_trial = _project_to_sphere(X_trial)
            e_trial = _log_energy(X_trial)
            if e_trial < cur_energy - 1e-15:
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            step *= 0.5
            if step < 1e-15:
                break
            continue
        
        X = X_trial
        cur_energy = e_trial
        if cur_energy < best_energy:
            best_energy = cur_energy
            best_X = X.copy()
        step = min(step * 1.5, 0.1)
    
    return best_X, best_energy


def construct():
    global _START
    _START = time.time()
    
    # Multiple diverse initializations
    inits = []
    inits.append(_golden_spiral(N))
    for seed in range(8):
        inits.append(_sobol_sphere(N, seed=seed))
    
    # Also add a few slightly perturbed golden spirals
    rng = np.random.default_rng(42)
    for k in range(4):
        pert = rng.standard_normal((N, 3)) * 0.1
        inits.append(_project_to_sphere(_golden_spiral(N) + pert))
    
    best_X = None
    best_energy = float('inf')
    
    # Optimize from each initialization
    for idx, X0 in enumerate(inits):
        if _remaining() < 10.0:
            break
        
        # Determine iterations based on remaining time
        # Estimate: each iteration is roughly O(N^2) = 10000 ops, fast
        # Budget remaining time evenly
        remaining_inits = len(inits) - idx
        time_per_init = max(1.0, _remaining() / max(1, remaining_inits))
        max_iter = int(time_per_init * 80)  # conservative estimate
        max_iter = max(200, min(max_iter, 3000))
        
        X_opt, e_opt = _optimize_from_start(X0, max_iter=max_iter, tol=1e-13)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_X = X_opt
    
    # Final high-precision polish on the best result
    if _remaining() > 10.0 and best_X is not None:
        # Run more iterations with tighter tolerance
        X_polish, e_polish = _optimize_from_start(best_X, max_iter=5000, tol=1e-15)
        if e_polish < best_energy:
            best_energy = e_polish
            best_X = X_polish
    
    if best_X is None:
        best_X = _golden_spiral(N)
    
    return _project_to_sphere(best_X)
