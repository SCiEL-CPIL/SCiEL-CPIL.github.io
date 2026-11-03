import numpy as np
import scipy.optimize
import time
import math

N = 100

def _elapsed(t0):
    return time.time() - t0

def _project_to_sphere(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _sobol_sphere(n):
    # Simple quasi-random using Halton-like sequence
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = []
    for i in range(1, n + 1):
        u = halton(i, 2)
        v = halton(i, 3)
        # Map to sphere using inverse CDF for uniform distribution
        z = 2.0 * u - 1.0  # uniform in [-1, 1]
        theta = 2.0 * math.pi * v
        r = math.sqrt(max(1.0 - z * z, 0.0))
        points.append([r * math.cos(theta), r * math.sin(theta), z])
    return np.array(points)

def _random_sphere(n, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    X = rng.randn(n, 3)
    return _project_to_sphere(X)

def _energy_precise(X):
    """Compute log energy with math.fsum for precision matching evaluator."""
    # Pairwise distances
    D = X @ X.T
    np.fill_diagonal(D, 0.0)
    dist_sq = 2.0 - 2.0 * D
    dist_sq = np.maximum(dist_sq, 1e-15)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(N, k=1)
    log_terms = np.log(dist[iu])
    return -math.fsum(log_terms.tolist())

def _energy(X):
    """Compute log energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)"""
    # Pairwise distances
    # d_ij^2 = 2 - 2*x_i.x_j
    D = X @ X.T
    np.fill_diagonal(D, 0.0)
    dist_sq = 2.0 - 2.0 * D
    dist_sq = np.maximum(dist_sq, 1e-15)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(N, k=1)
    return -np.sum(np.log(dist[iu]))

def _energy_and_grad(X):
    """Compute energy and gradient."""
    D = X @ X.T
    np.fill_diagonal(D, 0.0)
    dist_sq = 2.0 - 2.0 * D
    dist_sq = np.maximum(dist_sq, 1e-15)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(N, k=1)
    E = -np.sum(np.log(dist[iu]))
    
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    W_sum = W.sum(axis=1)
    WX = W @ X
    grad = -W_sum[:, np.newaxis] * X + WX
    
    return E, grad

def _spherical_to_cartesian(params):
    """Convert (theta_1..theta_N, phi_1..phi_N) to Cartesian on sphere."""
    theta = params[:N]
    phi = params[N:]
    # theta: polar angle [0, pi], phi: azimuthal [0, 2*pi]
    sin_theta = np.sin(theta)
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])

def _cartesian_to_spherical(X):
    """Convert Cartesian to (theta, phi) parameters."""
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    phi = np.where(phi < 0, phi + 2.0 * np.pi, phi)
    return np.concatenate([theta, phi])

def _objective_and_grad(params):
    """Objective and gradient in spherical coordinates."""
    X = _spherical_to_cartesian(params)
    E, grad_cart = _energy_and_grad(X)
    
    # Transform gradient from Cartesian to spherical coordinates
    # dE/dtheta_k = (dE/dx_k * dx_k/dtheta + dE/dy_k * dy_k/dtheta + dE/dz_k * dz_k/dtheta)
    # dx/dtheta = cos(theta)*cos(phi), dy/dtheta = cos(theta)*sin(phi), dz/dtheta = -sin(theta)
    # dE/dphi_k = (dE/dx_k * dx_k/dphi + dE/dy_k * dy_k/dphi + dE/dz_k * dz_k/dphi)
    # dx/dphi = -sin_theta*sin(phi), dy/dphi = sin_theta*cos(phi), dz/dphi = 0
    
    theta = params[:N]
    phi = params[N:]
    
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    
    gx = grad_cart[:, 0]
    gy = grad_cart[:, 1]
    gz = grad_cart[:, 2]
    
    grad_theta = gx * cos_theta * cos_phi + gy * cos_theta * sin_phi + gz * (-sin_theta)
    grad_phi = gx * (-sin_theta * sin_phi) + gy * (sin_theta * cos_phi) + gz * 0.0
    
    return E, np.concatenate([grad_theta, grad_phi])

def _optimize_spherical(X_init, t0, deadline, maxiter=200, tol=1e-12):
    """Optimize using L-BFGS-B in spherical coordinates."""
    params0 = _cartesian_to_spherical(X_init)
    
    # Bounds: theta in [0, pi], phi in [0, 2*pi]
    bounds = [(0.0, np.pi)] * N + [(0.0, 2.0 * np.pi)] * N
    
    result = scipy.optimize.minimize(
        _objective_and_grad,
        params0,
        jac=True,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': maxiter, 'ftol': tol, 'gtol': 1e-10, 'maxls': 50}
    )
    
    if result.success or result.fun < _energy(X_init):
        X_opt = _spherical_to_cartesian(result.x)
        X_opt = _project_to_sphere(X_opt)
        return X_opt, result.fun
    
    return X_init, _energy(X_init)

def _anneal(X, t0, deadline, n_restarts=10, noise_scale=0.5):
    """Simulated annealing style: perturb and re-optimize with adaptive noise."""
    best_X = X.copy()
    best_E = _energy(X)
    
    for i in range(n_restarts):
        if _elapsed(t0) > deadline - 10:
            break
        
        # Adaptive noise: start large, decrease geometrically
        scale = noise_scale * (0.7 ** i)
        noise = np.random.randn(N, 3) * scale
        X_pert = best_X + noise
        X_pert = _project_to_sphere(X_pert)
        
        # Use fewer iterations for early restarts, more for later
        maxiter = 50 if i < n_restarts // 2 else 150
        X_opt, E_opt = _optimize_spherical(X_pert, t0, deadline, maxiter=maxiter, tol=1e-10)
        
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt
    
    return best_X, best_E

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # 10s safety margin before 150s kill
    
    # Generate multiple diverse initial configurations
    inits = [
        _fibonacci_sphere(N),
        _sobol_sphere(N),
        _random_sphere(N, seed=42),
        _random_sphere(N, seed=123),
        _random_sphere(N, seed=777),
        _random_sphere(N, seed=2024),
    ]
    
    best_X = None
    best_E = float('inf')
    
    # Optimize each initial configuration
    for i, X_init in enumerate(inits):
        if _elapsed(t0) > deadline - 30:
            break
        
        X_opt, E_opt = _optimize_spherical(X_init, t0, deadline, maxiter=300, tol=1e-12)
        
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt
    
    # Multi-start refinement: perturb best and re-optimize
    if best_X is not None and _elapsed(t0) < deadline - 20:
        X_refined, E_refined = _anneal(best_X, t0, deadline, n_restarts=12, noise_scale=0.6)
        if E_refined < best_E:
            best_E = E_refined
            best_X = X_refined
    
    # Final high-precision polish with very tight tolerance
    if best_X is not None and _elapsed(t0) < deadline - 5:
        X_final, E_final = _optimize_spherical(best_X, t0, deadline, maxiter=1000, tol=1e-15)
        if E_final < best_E:
            best_E = E_final
            best_X = X_final
    
    # Ensure valid output
    best_X = _project_to_sphere(best_X)
    
    # Validate: no points closer than 1e-9
    D = best_X @ best_X.T
    np.fill_diagonal(D, 0.0)
    dist_sq = 2.0 - 2.0 * D
    min_dist_sq = np.min(dist_sq)
    if min_dist_sq < (1e-9) ** 2:
        # Fallback to Fibonacci if points are too close
        best_X = _fibonacci_sphere(N)
    
    return best_X
