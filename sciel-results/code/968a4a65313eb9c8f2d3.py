import numpy as np
import time
import math

N = 150

def _energy(points):
    """Compute logarithmic energy."""
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu[0], iu[1]]))

def _grad_and_energy(points):
    """Compute log energy and its gradient for points on unit sphere (vectorized)."""
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu[0], iu[1]]))
    
    coeff = np.zeros_like(dist_sq)
    mask = ~np.eye(n, dtype=bool)
    coeff[mask] = -1.0 / dist_sq[mask]
    
    row_sums = coeff.sum(axis=1)
    grad = points * row_sums[:, np.newaxis] - coeff @ points
    
    return energy, grad

def _project_to_sphere(points):
    """Project points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms

def _fibonacci_spiral(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed=None):
    """Generate random points on unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _riemannian_cg(points, max_iter=500, tol=1e-10, deadline=None):
    """Riemannian conjugate gradient on the sphere with backtracking line search."""
    n = points.shape[0]
    X = points.copy()
    e, g = _grad_and_energy(X)
    
    def _tangent(v, x):
        dot = np.sum(v * x, axis=1, keepdims=True)
        return v - dot * x
    
    g_t = _tangent(g, X)
    
    if np.linalg.norm(g_t) < tol:
        return X, e
    
    d = -g_t
    g_old_t = g_t
    g_old_norm_sq = np.sum(g_old_t * g_old_t)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Backtracking line search
        alpha = 1.0
        best_e = e
        best_X = X.copy()
        best_alpha = 1.0
        
        for _ls in range(20):
            X_new = X + alpha * d
            X_new = _project_to_sphere(X_new)
            
            e_new = _energy(X_new)
            if e_new < best_e - 1e-4 * alpha * np.sum(g_t * d):
                best_e = e_new
                best_alpha = alpha
                best_X = X_new
                break
            alpha *= 0.5
        
        if best_e >= e:
            # If no improvement, try smaller steps
            for alpha_try in [0.01, 0.001, 0.0001]:
                X_try = X + alpha_try * d
                X_try = _project_to_sphere(X_try)
                e_try = _energy(X_try)
                if e_try < best_e:
                    best_e = e_try
                    best_alpha = alpha_try
                    best_X = X_try
                    break
        
        X = best_X
        e = best_e
        
        e_new, g_new = _grad_and_energy(X)
        g_new_t = _tangent(g_new, X)
        g_new_norm_sq = np.sum(g_new_t * g_new_t)
        
        if np.sqrt(g_new_norm_sq) < tol:
            break
        
        # Fletcher-Reeves beta with safeguard
        if g_old_norm_sq > 1e-30:
            beta = g_new_norm_sq / g_old_norm_sq
        else:
            beta = 0.0
        beta = min(beta, 10.0)  # Cap beta to prevent divergence
        
        d = -g_new_t + beta * d
        d = _tangent(d, X)
        
        g_old_t = g_new_t
        g_old_norm_sq = g_new_norm_sq
    
    return X, e

def _lbfgs_optimize(points, max_iter=500, ftol=1e-14, gtol=1e-12, deadline=None):
    """Optimize using L-BFGS-B with projection."""
    from scipy.optimize import minimize
    
    def objective_and_grad(x_flat):
        if deadline is not None and time.time() > deadline:
            return np.inf, np.zeros_like(x_flat)
        pts = x_flat.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        e, g = _grad_and_energy(pts)
        dot = np.sum(g * pts, axis=1, keepdims=True)
        g_tangent = g - dot * pts
        return e, g_tangent.reshape(-1)
    
    try:
        result = minimize(
            objective_and_grad,
            points.reshape(-1),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
        )
        if result.x is not None and np.all(np.isfinite(result.x)):
            return _project_to_sphere(result.x.reshape(-1, 3))
    except Exception:
        pass
    
    return points

def _repulsion_refine(points, n_iter=500, deadline=None):
    """
    Vectorized repulsion-based refinement: move each point away from its neighbors.
    """
    X = points.copy()
    n = X.shape[0]
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Vectorized force computation
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (n, n, 3)
        dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
        np.fill_diagonal(dist_sq, 1.0)  # Avoid division by zero
        dist = np.sqrt(dist_sq)  # (n, n)
        
        # Force magnitude: 1/dist for log energy gradient
        # Force direction: diff / dist
        # Combined: diff / dist^2 = diff * (1/dist^2)
        # But we want unit direction * magnitude = (diff/dist) * (1/dist) = diff/dist^2
        inv_dist_sq = 1.0 / dist_sq  # (n, n)
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        # forces[i] = sum_j (X[i] - X[j]) / |X[i] - X[j]|^2
        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)  # (n, 3)
        
        # Project to tangent space and take step
        dot = np.sum(forces * X, axis=1, keepdims=True)  # (n, 1)
        f_tan = forces - dot * X  # (n, 3)
        
        X = X + 0.01 * f_tan
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        X = X / norms
    
    return X

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    # Generate multiple initial configurations
    inits = []
    inits.append(_fibonacci_spiral(N))
    for s in range(15):
        inits.append(_random_points(N, seed=42 + s))
    
    best_points = None
    best_energy = np.inf
    
    # Phase 1: L-BFGS on all initial configurations
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 60.0:
            break
        
        try:
            cand = _lbfgs_optimize(init_pts, max_iter=300, ftol=1e-10, gtol=1e-8, deadline=deadline)
            cand_energy = _energy(cand)
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_points = cand.copy()
        except Exception:
            pass
    
    if best_points is None:
        best_points = _fibonacci_spiral(N)
        best_energy = _energy(best_points)
    
    # Phase 2: Riemannian CG polish
    if time.time() < deadline - 30.0:
        try:
            cand, cand_energy = _riemannian_cg(best_points, max_iter=1000, tol=1e-12, deadline=deadline)
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_points = cand.copy()
        except Exception:
            pass
    
    # Phase 3: L-BFGS polish with tighter tolerances
    if time.time() < deadline - 15.0:
        try:
            cand = _lbfgs_optimize(best_points, max_iter=1000, ftol=1e-14, gtol=1e-12, deadline=deadline)
            cand_energy = _energy(cand)
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_points = cand.copy()
        except Exception:
            pass
    
    # Phase 4: Final Riemannian CG polish
    if time.time() < deadline - 5.0:
        try:
            cand, cand_energy = _riemannian_cg(best_points, max_iter=2000, tol=1e-14, deadline=deadline)
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_points = cand.copy()
        except Exception:
            pass
    
    # Phase 5: Final L-BFGS with very tight tolerances
    if time.time() < deadline - 2.0:
        try:
            cand = _lbfgs_optimize(best_points, max_iter=2000, ftol=1e-16, gtol=1e-14, deadline=deadline)
            cand_energy = _energy(cand)
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_points = cand.copy()
        except Exception:
            pass
    
    # Phase 6: Very final micro-polish with Riemannian CG
    if time.time() < deadline - 1.0:
        try:
            cand, cand_energy = _riemannian_cg(best_points, max_iter=5000, tol=1e-15, deadline=deadline)
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_points = cand.copy()
        except Exception:
            pass
    
    # Final projection and validation
    best_points = _project_to_sphere(best_points)
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                best_points[j] += 1e-8 * np.random.randn(3)
                best_points = _project_to_sphere(best_points)
    
    return best_points
