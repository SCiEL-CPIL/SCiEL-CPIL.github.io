import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _energy(X):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    eps = 1e-12
    safe_dists = np.maximum(dists, eps)
    log_dists = np.log(safe_dists)
    np.fill_diagonal(log_dists, 0.0)
    # sum over i<j: use upper triangle
    iu = np.triu_indices(N, k=1)
    return -np.sum(log_dists[iu])

def _riemannian_grad(X):
    """
    Riemannian gradient of the logarithmic energy on the unit sphere.
    For energy E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij),
    the Euclidean gradient w.r.t. x_i is: sum_{j != i} (x_i - x_j) / d_ij^2
    The Riemannian gradient is the projection onto the tangent space:
    grad_R(x_i) = grad_E(x_i) - (grad_E(x_i) . x_i) * x_i
    """
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    eps = 1e-12
    safe_dists = np.maximum(dists, eps)
    
    # Euclidean gradient: grad_i = sum_{j != i} (x_i - x_j) / d_ij^2
    inv_dists_sq = 1.0 / (safe_dists ** 2)
    np.fill_diagonal(inv_dists_sq, 0.0)
    
    euclid_grad = np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)
    
    # Project to tangent space: remove component along x_i
    proj_coeff = np.sum(euclid_grad * X, axis=1, keepdims=True)
    rgrad = euclid_grad - proj_coeff * X
    
    return rgrad

def _retract(X, V, step):
    """
    Retraction on the sphere: move along the tangent direction V by step, then normalize.
    This is the standard retraction: R_x(v) = x + v, then normalize.
    """
    X_new = X + step * V
    norms = np.linalg.norm(X_new, axis=1, keepdims=True)
    # Avoid zero norms
    norms = np.maximum(norms, 1e-15)
    X_new = X_new / norms
    return X_new

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng):
    v = rng.normal(size=(n, 3))
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return v

def _gaussian_noise_sphere(X, sigma, rng):
    """Add Gaussian noise and renormalize."""
    noise = rng.normal(0, sigma, X.shape)
    X_new = X + noise
    norms = np.linalg.norm(X_new, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X_new / norms

def _lbfgs_polish(X0, max_iter=200, ftol=1e-12, gtol=1e-10):
    """
    Polish using L-BFGS-B with sphere constraint handled via penalty.
    """
    def obj_and_grad(flat):
        X = flat.reshape(N, 3)
        # Normalize to sphere for energy computation
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        Xn = X / norms
        
        # Compute Riemannian gradient
        rgrad = _riemannian_grad(Xn)
        
        # Energy
        E = _energy(Xn)
        
        # Penalty for leaving the sphere
        lambda_pen = 1000.0
        norms_sq = np.sum(X**2, axis=1)
        penalty = 0.5 * lambda_pen * np.sum((norms_sq - 1.0)**2)
        pen_grad = lambda_pen * (norms_sq - 1.0)[:, np.newaxis] * X
        
        total_E = E + penalty
        # The gradient w.r.t. flat X: project Riemannian grad back to Euclidean
        # Actually, for L-BFGS-B, we need the Euclidean gradient of the penalized objective.
        # The Riemannian gradient is the projection of the Euclidean gradient.
        # So Euclidean grad = rgrad + (rgrad . x_i) x_i + pen_grad
        # But since rgrad is already projected, and we're on the sphere, 
        # the Euclidean gradient of E restricted to sphere is rgrad.
        # For the penalized objective, we add pen_grad.
        # However, the Euclidean gradient of E (unconstrained) is:
        # sum_{j != i} (x_i - x_j) / d_ij^2
        # Let's compute it directly to be safe.
        diff = Xn[:, np.newaxis, :] - Xn[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1.0)
        eps = 1e-12
        safe_dists = np.maximum(dists, eps)
        inv_dists_sq = 1.0 / (safe_dists ** 2)
        np.fill_diagonal(inv_dists_sq, 0.0)
        euclid_grad = np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)
        
        total_grad = euclid_grad + pen_grad
        
        return total_E, total_grad.flatten()
    
    flat0 = X0.flatten()
    res = minimize(
        obj_and_grad,
        flat0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
    )
    X_opt = res.x.reshape(N, 3)
    # Normalize
    norms = np.linalg.norm(X_opt, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    X_opt = X_opt / norms
    return X_opt, _energy(X_opt)

def _anneal_and_optimize(X0, time_budget, rng):
    """
    Adaptive simulated annealing:
    1. Start with X0
    2. Iterate: compute Riemannian gradient, take a step, occasionally perturb
    3. Periodically polish with L-BFGS-B
    """
    start_time = time.time()
    X = X0.copy()
    best_X = X.copy()
    best_E = _energy(X)
    
    # Annealing parameters
    T0 = 0.5  # initial temperature (step size scale)
    T_end = 0.01
    cooling_rate = 0.995
    
    T = T0
    step = 0.01
    
    while time.time() - start_time < time_budget:
        remaining = time_budget - (time.time() - start_time)
        if remaining < 2.0:
            break
        
        # Compute Riemannian gradient
        rgrad = _riemannian_grad(X)
        grad_norm = np.linalg.norm(rgrad)
        
        if grad_norm < 1e-12:
            break
        
        # Adaptive step size: scale by temperature
        # Use a line search-like approach: try the step, if energy increases too much, reduce
        effective_step = step * T
        
        # Take a Riemannian gradient step
        X_new = _retract(X, rgrad, -effective_step)  # negative gradient
        E_new = _energy(X_new)
        
        # Accept if energy decreased, or with probability exp(-(E_new - E)/T) if increased
        if E_new < best_E:
            best_E = E_new
            best_X = X_new.copy()
        
        delta_E = E_new - _energy(X)
        if delta_E < 0 or (T > 0 and np.random.random() < np.exp(-delta_E / max(T, 1e-10))):
            X = X_new
        else:
            # Try a smaller step
            X_new2 = _retract(X, rgrad, -effective_step * 0.5)
            E_new2 = _energy(X_new2)
            if E_new2 < _energy(X):
                X = X_new2
                if E_new2 < best_E:
                    best_E = E_new2
                    best_X = X_new2.copy()
        
        # Occasionally perturb to escape local minima
        if rng.random() < 0.05:
            X = _gaussian_noise_sphere(X, T * 0.1, rng)
            E_pert = _energy(X)
            if E_pert < best_E:
                best_E = E_pert
                best_X = X.copy()
        
        # Cool down
        T = max(T * cooling_rate, T_end)
        
        # Periodically polish with L-BFGS-B
        if time.time() - start_time > 5.0 and (int((time.time() - start_time) // 10) > 0):
            if time.time() - start_time % 10 < 1.0:  # rough check
                pass
        
        # Every ~5 seconds, polish
        elapsed = time.time() - start_time
        if elapsed > 5.0 and int(elapsed // 5) > int((elapsed - 1.0) // 5):
            if remaining > 3.0:
                X_polished, E_polished = _lbfgs_polish(X, max_iter=100, ftol=1e-10, gtol=1e-8)
                if E_polished < best_E:
                    best_E = E_polished
                    best_X = X_polished.copy()
                X = X_polished
    
    return best_X, best_E

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s
    
    rng = np.random.default_rng(12345)
    
    best_X = None
    best_energy = float('inf')
    
    # Generate diverse initializers
    initializers = []
    
    # 1. Fibonacci sphere
    initializers.append(_fibonacci_sphere(N))
    
    # 2. Random points
    for _ in range(3):
        initializers.append(_random_sphere(N, rng))
    
    # 3. Fibonacci + various noise levels
    X_fib = _fibonacci_sphere(N)
    for sigma in [0.05, 0.1, 0.2, 0.5]:
        initializers.append(_gaussian_noise_sphere(X_fib, sigma, rng))
    
    # 4. Clustered initializers (hemispheres)
    X_hemi = np.zeros((N, 3))
    for i in range(N):
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v)
        if i % 2 == 0:
            v[2] = np.abs(v[2]) + 0.3
        else:
            v[2] = -np.abs(v[2]) - 0.3
        v = v / np.linalg.norm(v)
        X_hemi[i] = v
    initializers.append(X_hemi)
    
    # 5. Uniform grid on sphere (approximate)
    # Use spherical coordinates with uniform area distribution
    theta = np.linspace(0, np.pi, N)
    phi = np.linspace(0, 2*np.pi, N)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    X_grid = np.column_stack([
        np.sin(TH) * np.cos(PH).ravel(),
        np.sin(TH) * np.sin(PH).ravel(),
        np.cos(TH).ravel()
    ])
    initializers.append(X_grid)
    
    # Sort initializers by initial energy (best first)
    initializer_energies = []
    for X0 in initializers:
        E0 = _energy(X0)
        initializer_energies.append((E0, X0))
    initializer_energies.sort(key=lambda x: x[0])
    
    # Allocate time: give more time to better initializers
    total_initializers = len(initializer_energies)
    base_time = 10.0
    bonus_time = 5.0
    
    for idx, (E0, X0) in enumerate(initializer_energies):
        if time.time() > deadline - 10.0:
            break
        
        remaining = deadline - time.time() - 5.0
        if remaining < 3.0:
            break
        
        # Allocate time based on rank: better initializers get more time
        rank_factor = 1.0 - 0.3 * (idx / max(total_initializers - 1, 1))
        time_budget = min(base_time * rank_factor + bonus_time * (1.0 - idx / max(total_initializers, 1)), remaining)
        time_budget = max(time_budget, 3.0)
        
        X_opt, energy = _anneal_and_optimize(X0, time_budget, rng)
        
        if energy < best_energy:
            best_energy = energy
            best_X = X_opt
    
    # Final polish of the best result
    if best_X is not None and time.time() < deadline - 15.0:
        remaining = deadline - time.time() - 10.0
        if remaining > 5.0:
            X_polished, E_polished = _lbfgs_polish(best_X, max_iter=500, ftol=1e-12, gtol=1e-10)
            if E_polished < best_energy:
                best_energy = E_polished
                best_X = X_polished
    
    # Final safety polish
    if best_X is not None and time.time() < deadline - 5.0:
        remaining = deadline - time.time() - 2.0
        if remaining > 3.0:
            X_final, E_final = _lbfgs_polish(best_X, max_iter=200, ftol=1e-12, gtol=1e-10)
            if E_final < best_energy:
                best_energy = E_final
                best_X = X_final
    
    # Final normalization and validation
    if best_X is None:
        best_X = _fibonacci_sphere(N)
    else:
        norms = np.linalg.norm(best_X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        best_X = best_X / norms
    
    # Validate: no points closer than 1e-9
    diff = best_X[:, np.newaxis, :] - best_X[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    min_dist = np.min(dists)
    if min_dist < 1e-9:
        # Fallback to Fibonacci
        best_X = _fibonacci_sphere(N)
    
    return best_X
