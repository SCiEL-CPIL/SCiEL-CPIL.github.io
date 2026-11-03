"""Optimized logarithmic energy minimization for 100 points on the unit sphere.
Uses a multi-stage strategy:
1. Low-dimensional optimization of a generalized Fibonacci spiral.
2. Riemannian gradient descent for broad convergence.
3. High-precision polish using trust-constr with explicit sphere constraints.
"""
import numpy as np
from scipy.optimize import minimize
import time
import math

N = 100
DEADLINE = 145.0  # Safety margin before 150s kill


def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _energy_and_grad(points):
    """Compute logarithmic energy and gradient for points on unit sphere."""
    n = points.shape[0]
    sq_norms = np.einsum('ij,ij->i', points, points)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (points @ points.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-30)
    
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(np.log(dist_sq[iu]))
    
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    sum_W = W.sum(axis=1, keepdims=True)
    Wx = W @ points
    raw_grad = sum_W * points - Wx
    
    grad = -raw_grad
    
    dot = np.sum(grad * points, axis=1, keepdims=True)
    grad = grad - dot * points
    
    return energy, grad


def _energy_only(points):
    """Just the energy value (no gradient)."""
    n = points.shape[0]
    sq_norms = np.einsum('ij,ij->i', points, points)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (points @ points.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-30)
    iu = np.triu_indices(n, k=1)
    return -0.5 * np.sum(np.log(dist_sq[iu]))


def _precise_energy(pts):
    """Compute energy using math.fsum for accuracy."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _exponential_map(points, tangent_vecs):
    """Riemannian exponential map on the sphere."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms_safe = np.where(norms < 1e-15, 1.0, norms)
    unit_tangent = tangent_vecs / norms_safe
    cos_t = np.cos(norms)
    sin_t = np.sin(norms)
    new_points = points * cos_t + unit_tangent * sin_t
    return _normalize(new_points)


def _riemannian_gradient_descent(points, deadline, max_iter=5000, lr_init=0.01):
    """Riemannian gradient descent with backtracking line search."""
    current = points.copy()
    best_e = _energy_only(current)
    best_pts = current.copy()
    lr = lr_init
    c1 = 0.1
    rho = 0.5
    no_improve = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, g = _energy_and_grad(current)
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-12:
            break
        
        step = lr
        found = False
        for bt in range(15):
            tangent = -step * g
            new_pts = _exponential_map(current, tangent)
            new_e = _energy_only(new_pts)
            if new_e <= e + c1 * step * g_norm * g_norm:
                found = True
                break
            step *= rho
        
        if not found:
            step = max(lr * rho**15, 1e-15)
            tangent = -step * g
            new_pts = _exponential_map(current, tangent)
            new_e = _energy_only(new_pts)
        
        if new_e < best_e - 1e-14:
            current = new_pts
            best_e = new_e
            best_pts = current.copy()
            no_improve = 0
            lr = min(step * 1.3, 0.05)
        elif new_e < best_e:
            current = new_pts
            no_improve += 1
            lr = step
        else:
            lr = step * 0.5
            no_improve += 1
            if lr < 1e-14 or no_improve > 40:
                break
    
    return best_pts, best_e


def _generalized_spiral(params):
    """
    Generate points on a generalized spiral.
    params: [alpha, beta, gamma]
    z_k = 1 - 2 * (k + alpha) / N
    phi_k = beta * k + gamma * k^2
    """
    alpha, beta, gamma = params
    k = np.arange(N)
    z = 1.0 - 2.0 * (k + alpha) / N
    # Clip z to [-1, 1] to avoid sqrt of negative if alpha is out of bounds
    z = np.clip(z, -1.0, 1.0)
    phi = beta * k + gamma * k * k
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)


def _optimize_spiral_params(init_params, deadline):
    """Optimize the parameters of the generalized spiral."""
    def objective(params):
        pts = _generalized_spiral(params)
        return _energy_only(pts)
    
    try:
        result = minimize(
            objective,
            init_params,
            method='L-BFGS-B',
            bounds=[(-1.0, 2.0), (2.0, 5.0), (-0.1, 0.1)],
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return result.x
    except Exception:
        return init_params


def _trust_constr_polish(pts, deadline):
    """Polish using trust-constr with explicit sphere constraints."""
    n = pts.shape[0]
    x0 = pts.reshape(-1)
    
    def objective(flat):
        p = flat.reshape(n, 3)
        # Soft projection to avoid numerical issues during optimization
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        p = p / norms
        return _energy_only(p)
    
    def gradient(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        p = p / norms
        _, g = _energy_and_grad(p)
        return g.reshape(-1)
    
    # Define constraints: ||x_i||^2 - 1 = 0 for each i
    constraints = []
    for i in range(n):
        def constr(flat, i=i):
            return flat[i*3:(i+1)*3] @ flat[i*3:(i+1)*3] - 1.0
        constraints.append({'type': 'eq', 'fun': constr})
        
    # To speed up, we might not want 100 constraints. 
    # Instead, we can use a penalty method or just rely on the projection in the objective.
    # However, trust-constr handles constraints well. 
    # For 100 points, 100 constraints is manageable.
    
    try:
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method='trust-constr',
            constraints=constraints,
            options={'maxiter': 100, 'gtol': 1e-10, 'xtol': 1e-12}
        )
        new_pts = _normalize(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_energy = float('inf')
    
    # Phase 1: Optimize Generalized Spirals
    # Try a few different initial parameter sets
    init_params_list = [
        [0.5, math.pi * (1 + np.sqrt(5)), 0.0], # Standard Fibonacci
        [0.1, math.pi * (1 + np.sqrt(5)), 0.01],
        [0.9, math.pi * (1 + np.sqrt(5)), -0.01],
        [0.5, math.pi * (1 + np.sqrt(5)) * 0.9, 0.0],
        [0.5, math.pi * (1 + np.sqrt(5)) * 1.1, 0.0],
    ]
    
    spiral_candidates = []
    for init_params in init_params_list:
        if time.time() > deadline - 60.0:
            break
        opt_params = _optimize_spiral_params(init_params, deadline - 50.0)
        pts = _generalized_spiral(opt_params)
        e = _energy_only(pts)
        spiral_candidates.append((e, pts))
        
    spiral_candidates.sort(key=lambda x: x[0])
    
    # Phase 2: Riemannian Gradient Descent from best spirals
    # Use the top 3 spirals
    for e, pts in spiral_candidates[:3]:
        if time.time() > deadline - 30.0:
            break
        pts_rgd, e_rgd = _riemannian_gradient_descent(pts, deadline - 10.0, max_iter=3000)
        if e_rgd < best_energy:
            best_energy = e_rgd
            best_points = pts_rgd.copy()
            
    # If no good candidate, fallback to random
    if best_points is None:
        pts = _generalized_spiral([0.5, math.pi * (1 + np.sqrt(5)), 0.0])
        best_points = pts
        best_energy = _energy_only(pts)
        
    # Phase 3: High-Precision Polish with Trust-Constr
    if time.time() < deadline - 5.0:
        pts_tc = _trust_constr_polish(best_points, deadline - 2.0)
        e_tc = _energy_only(pts_tc)
        if e_tc < best_energy:
            best_energy = e_tc
            best_points = pts_tc.copy()
            
    # Final Validation
    best_points = _normalize(best_points)
    
    # Check for any points too close together
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                best_points[j] = best_points[j] + 1e-8 * rng.standard_normal(3)
                best_points = _normalize(best_points)
    
    if not np.all(np.isfinite(best_points)):
        best_points = _generalized_spiral([0.5, math.pi * (1 + np.sqrt(5)), 0.0])
        
    return best_points
