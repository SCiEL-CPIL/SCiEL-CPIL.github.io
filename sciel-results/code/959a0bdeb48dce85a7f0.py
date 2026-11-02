"""Optimized logarithmic energy of 150 points on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE = 145.0  # safety margin before 150s kill

def _log_energy(X):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = X.shape[0]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    dist = np.sqrt(dist2)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu]))

def _energy_and_grad(X):
    """Compute logarithmic energy and Riemannian gradient for points on the unit sphere."""
    n = X.shape[0]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    dist = np.sqrt(dist2)
    
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Euclidean gradient: grad_eucl[i] = sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2
    eucl_grad = -np.sum(diff / dist2[:, :, np.newaxis], axis=1)
    
    # Project to tangent space
    dot = np.sum(eucl_grad * X, axis=1, keepdims=True)
    riem_grad = eucl_grad - dot * X
    
    return energy, riem_grad

def _project_to_sphere(X):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms

def _exponential_map(X, V):
    """Exponential map on the sphere: move from X in direction V (tangent)."""
    theta = np.linalg.norm(V, axis=1, keepdims=True)
    theta = np.maximum(theta, 1e-15)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    # For small theta, use series expansion
    small = theta < 1e-8
    if np.any(small):
        V_norm = V[small] / theta[small]
        X_new_small = X[small] * cos_theta[small] + V_norm * sin_theta[small]
    else:
        V_norm = V / theta
        X_new = X * cos_theta + V_norm * sin_theta
        if np.any(small):
            X_new[small] = X_new_small
        return _project_to_sphere(X_new)
    return _project_to_sphere(X)

def _fibonacci_init(n):
    """Golden-angle Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _halton_init(n, base=2):
    """Halton sequence on the sphere."""
    points = []
    for i in range(1, n + 1):
        # 2D Halton
        f2, f3 = 0.0, 0.0
        p2, p3 = 2, 3
        i2, i3 = i, i
        while i2 > 0:
            i2 //= p2
            f2 += (i2 % p2) / (p2 ** (i2 // p2 + 1)) if i2 > 0 else 0
            break
        # Simpler: use standard Halton
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        u1 = halton(i, 2)
        u2 = halton(i, 3)
        
        # Map to sphere using spherical coordinates
        theta = 2 * np.pi * u1
        phi = np.arccos(1 - 2 * u2)
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        points.append([x, y, z])
    
    return np.array(points)

def _farthest_point_init(n, seed=42):
    """Farthest point sampling initialization."""
    rng = np.random.RandomState(seed)
    # Start with two antipodal points
    X = np.array([[1, 0, 0], [-1, 0, 0]], dtype=float)
    
    while len(X) < n:
        # Compute min distance from each unplaced point candidate to existing points
        # Use a grid of candidates
        n_theta = 50
        n_phi = 50
        thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        phis = np.linspace(0, np.pi, n_phi)
        
        candidates = []
        for th in thetas:
            for ph in phis:
                pt = np.array([np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th), np.cos(ph)])
                candidates.append(pt)
        candidates = np.array(candidates)
        
        # Find candidate farthest from all existing points
        min_dists = np.inf * np.ones(len(candidates))
        for pt in X:
            d = np.linalg.norm(candidates - pt, axis=1)
            min_dists = np.minimum(min_dists, d)
        
        idx = np.argmax(min_dists)
        X = np.vstack([X, candidates[idx]])
    
    return X[:n]

def _riemannian_cg(X0, max_iter=200, time_limit=10.0):
    """Riemannian Conjugate Gradient method."""
    start_time = time.time()
    X = X0.copy()
    energy, grad = _energy_and_grad(X)
    
    # Initial search direction
    d = -grad
    beta = 0.0
    
    for k in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Line search: minimize f(X + alpha * exp_X(alpha*d))
        # Use backtracking
        alpha = 1.0
        c1 = 1e-4
        
        for ls_iter in range(30):
            X_trial = _exponential_map(X, alpha * d)
            energy_trial, grad_trial = _energy_and_grad(X_trial)
            
            # Check Armijo condition
            if energy_trial <= energy + c1 * alpha * np.sum(grad * d):
                break
            alpha *= 0.5
        
        if alpha < 1e-15:
            break
        
        # Update
        X = X_trial
        energy = energy_trial
        grad_new = grad_trial
        
        # Compute beta (Polak-Ribière)
        beta_new = max(0.0, np.sum(grad_new * (grad_new - grad)) / max(np.sum(grad * grad), 1e-15))
        d = -grad_new + beta_new * d
        
        # Project d to tangent space
        dot = np.sum(d * X, axis=1, keepdims=True)
        d = d - dot * X
        
        grad = grad_new
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-10:
            break
        
        # Restart if beta is too large
        if beta_new > 1.0:
            beta = 0.0
            d = -grad
    
    return X, energy

def _lbfgs_polish(X0, max_iter=1000, time_limit=15.0):
    """Polish using L-BFGS-B in R^3 with projection."""
    start_time = time.time()
    
    def objective(x_flat):
        X = x_flat.reshape(N, 3)
        X = _project_to_sphere(X)
        return _log_energy(X)
    
    def gradient(x_flat):
        X = x_flat.reshape(N, 3)
        X = _project_to_sphere(X)
        _, g = _energy_and_grad(X)
        return g.flatten()
    
    try:
        result = minimize(
            objective,
            X0.flatten(),
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'm': 20}
        )
        X_opt = _project_to_sphere(result.x.reshape(N, 3))
        e_opt = _log_energy(X_opt)
        return X_opt, e_opt
    except Exception:
        return X0, _log_energy(X0)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    initializations = []
    
    # 1. Fibonacci spiral
    initializations.append(_fibonacci_init(N))
    
    # 2. Halton sequence
    initializations.append(_halton_init(N))
    
    # 3. Farthest point sampling
    initializations.append(_farthest_point_init(N, seed=42))
    
    # 4. Random with fixed seed
    rng = np.random.RandomState(123)
    rand_pts = rng.randn(N, 3)
    initializations.append(_project_to_sphere(rand_pts))
    
    # 5. Another random seed
    rng2 = np.random.RandomState(456)
    rand_pts2 = rng2.randn(N, 3)
    initializations.append(_project_to_sphere(rand_pts2))
    
    # 6. Fibonacci with perturbation
    fib_perturbed = _fibonacci_init(N).copy()
    rng4 = np.random.RandomState(321)
    fib_perturbed += rng4.normal(0, 0.05, size=fib_perturbed.shape)
    initializations.append(_project_to_sphere(fib_perturbed))
    
    best_energy = np.inf
    best_points = initializations[0]
    
    # Phase 1: Riemannian CG from multiple starts
    for init_idx, X0 in enumerate(initializations):
        if time.time() > deadline - 40.0:
            break
        
        time_budget = (deadline - 40.0 - time.time()) / max(1, len(initializations) - init_idx)
        time_budget = max(time_budget, 5.0)
        
        try:
            X_opt, e_opt = _riemannian_cg(X0, max_iter=200, time_limit=time_budget)
            
            if time.time() < deadline - 20.0:
                X_polished, e_polished = _lbfgs_polish(X_opt, max_iter=500, time_limit=15.0)
                
                if e_polished < e_opt:
                    X_opt = X_polished
                    e_opt = e_polished
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_points = X_opt.copy()
                
        except Exception:
            pass

    # Phase 2: Final high-precision polish
    if time.time() < deadline - 5.0:
        try:
            X_opt, e_opt = _lbfgs_polish(best_points, max_iter=2000, time_limit=30.0)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_points = X_opt
            
            if time.time() < deadline - 3.0:
                X_opt2, e_opt2 = _lbfgs_polish(best_points, max_iter=1000, time_limit=10.0)
                if e_opt2 < best_energy:
                    best_energy = e_opt2
                    best_points = X_opt2
        except Exception:
            pass
    
    best_points = _project_to_sphere(best_points)
    
    # Check for points too close together
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(N, k=1)
    min_dist = np.sqrt(dist2[iu]).min()
    
    if min_dist < 1e-9:
        return _fibonacci_init(N)
    
    return best_points
