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
        z = 2.0 * u - 1.0
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

def _energy(X):
    D = X @ X.T
    np.fill_diagonal(D, 0.0)
    dist_sq = 2.0 - 2.0 * D
    dist_sq = np.maximum(dist_sq, 1e-15)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    return -np.sum(np.log(dist[iu]))

def _spherical_to_cartesian(params):
    theta = params[:N]
    phi = params[N:]
    sin_theta = np.sin(theta)
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])

def _cartesian_to_spherical(X):
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    phi = np.where(phi < 0, phi + 2.0 * np.pi, phi)
    return np.concatenate([theta, phi])

def _objective_and_grad(params):
    X = _spherical_to_cartesian(params)
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
    grad_cart = -W_sum[:, np.newaxis] * X + WX
    
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
    grad_phi = gx * (-sin_theta * sin_phi) + gy * (sin_theta * cos_phi)
    
    return E, np.concatenate([grad_theta, grad_phi])

def _local_minimize(params0, t0, deadline):
    bounds = [(0.0, np.pi)] * N + [(0.0, 2.0 * np.pi)] * N
    try:
        result = scipy.optimize.minimize(
            _objective_and_grad,
            params0,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
        )
        if _elapsed(t0) > deadline - 2:
             return params0, _objective_and_grad(params0)[0]
        return result.x, result.fun
    except Exception:
        return params0, _objective_and_grad(params0)[0]

def _basinhopping_run(X_init, t0, deadline, n_iter=20):
    params0 = _cartesian_to_spherical(X_init)
    current_energy = _objective_and_grad(params0)[0]
    
    def kick(x):
        # Random perturbation
        stepsize = 0.1
        x_new = x + np.random.randn(len(x)) * stepsize
        # Wrap angles
        x_new[:N] = np.clip(x_new[:N], 0, np.pi)
        x_new[N:] = np.mod(x_new[N:], 2.0 * np.pi)
        return x_new

    # We implement a simple basinhopping loop to control time
    best_x = params0.copy()
    best_E = current_energy
    
    for i in range(n_iter):
        if _elapsed(t0) > deadline - 5:
            break
        
        # Local minimize
        x_opt, E_opt = _local_minimize(best_x, t0, deadline)
        
        if E_opt < best_E:
            best_E = E_opt
            best_x = x_opt
        
        # Kick
        x_kick = kick(best_x)
        
        # Local minimize from kick
        x_opt2, E_opt2 = _local_minimize(x_kick, t0, deadline)
        
        if E_opt2 < best_E:
            best_E = E_opt2
            best_x = x_opt2
        else:
            # Accept with probability if worse (Metropolis)
            # T decreases over time
            T = 1.0 / (1.0 + i)
            if np.random.rand() < np.exp(-(E_opt2 - best_E) / T):
                best_x = x_opt2
                best_E = E_opt2

    return _spherical_to_cartesian(best_x), best_E

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # 5s safety margin
    
    inits = [
        _fibonacci_sphere(N),
        _sobol_sphere(N),
        _random_sphere(N, seed=42),
        _random_sphere(N, seed=123),
    ]
    
    best_X = None
    best_E = float('inf')
    
    for i, X_init in enumerate(inits):
        if _elapsed(t0) > deadline - 30:
            break
        
        # Run basinhopping
        X_opt, E_opt = _basinhopping_run(X_init, t0, deadline, n_iter=10)
        
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt
            
    # Final polish
    if best_X is not None and _elapsed(t0) < deadline - 5:
        params_best = _cartesian_to_spherical(best_X)
        X_final, E_final = _local_minimize(params_best, t0, deadline)
        if E_final < best_E:
            best_E = E_final
            best_X = X_final

    best_X = _project_to_sphere(best_X)
    
    # Validate
    D = best_X @ best_X.T
    np.fill_diagonal(D, 0.0)
    dist_sq = 2.0 - 2.0 * D
    min_dist_sq = np.min(dist_sq)
    if min_dist_sq < (1e-9) ** 2:
        best_X = _fibonacci_sphere(N)

    return best_X
