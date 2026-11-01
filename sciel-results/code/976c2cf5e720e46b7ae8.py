"""Stronger algorithm for 100-point logarithmic energy on the unit sphere.
Uses Riemannian exponential map gradient flow with aggressive L-BFGS-B polish.
Optimized for convergence quality over breadth of restarts.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = time.time() + 145.0  # Hard wall-clock deadline with 5s safety margin


def _time_left():
    return max(0.0, DEADLINE - time.time())


def _energy_fsum(points):
    """Compute logarithmic energy using math.fsum for maximum precision."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d = dist[mask]
    d = np.maximum(d, 1e-15)
    return math.fsum(np.log(1.0 / d).tolist())


def _energy_grad_vec(points):
    """Vectorized energy (np.sum, fast) and gradient on the sphere.
    E = sum_{i<j} log(1/||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    dE/dx_i = -sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    """
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy (fast, for comparison during optimization)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_sq = dist_sq[mask]
    d_sq = np.maximum(d_sq, 1e-30)
    energy = -0.5 * np.sum(np.log(d_sq))
    
    # Gradient
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.einsum('ij,ijk->ik', inv_dist_sq, diff)
    
    # Project gradient to tangent space
    g_tangent = grad - (grad * points).sum(axis=1, keepdims=True) * points
    
    return energy, g_tangent


def _project(points):
    """Project to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _exp_map(points, tangent_vecs):
    """Exponential map on the unit sphere: move along geodesic.
    For unit sphere, exp_x(v) = x*cos(||v||) + v/sin(||v||) * sin(||v||)
    Simplifies to: x*cos(r) + (v/r)*sin(r) where r=||v||
    """
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    # Handle zero vectors
    safe_norms = np.maximum(norms, 1e-15)
    cos_r = np.cos(norms)
    sin_r = np.sin(safe_norms)
    # exp_x(v) = x * cos(||v||) + (v / ||v||) * sin(||v||)
    result = points * cos_r + tangent_vecs / safe_norms * sin_r
    return _project(result)


def _riemannian_flow(points, time_budget, initial_lr=0.05, min_lr=1e-7,
                     lr_decay=0.999, patience=2000):
    """Riemannian gradient descent using exponential map with backtracking line search.
    Operates directly on the sphere manifold via geodesic steps.
    """
    start_time = time.time()
    best_points = points.copy()
    best_energy, _ = _energy_grad_vec(best_points)
    lr = initial_lr
    no_improve = 0
    
    while time.time() - start_time < time_budget:
        energy, grad = _energy_grad_vec(best_points)
        
        # Check if gradient is small enough
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        # Backtracking line search with exponential map
        alpha = lr
        improved = False
        for _ in range(25):
            # Riemannian step: move along geodesic in direction -grad
            step = -alpha * grad
            new_points = _exp_map(best_points, step)
            
            new_energy, _ = _energy_grad_vec(new_points)
            
            if new_energy < energy - 1e-12:
                best_points = new_points
                best_energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            lr *= 0.85
            if no_improve > patience:
                break
    
    return best_points


def _polish(points, time_budget):
    """High-precision polish using L-BFGS-B with very tight tolerances.
    Multiple passes with progressively tighter tolerances."""
    start_time = time.time()
    current = points.copy()
    
    for pass_num in range(3):
        if _time_left() < 3.0:
            break
        
        def fun_and_grad(x):
            p = _project(x.reshape(-1, 3))
            e, g = _energy_grad_vec(p)
            return e, g.flatten()
        
        x0 = current.flatten().copy()
        
        try:
            if pass_num == 0:
                opts = {'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-11, 'maxls': 100, 'm': 50}
            elif pass_num == 1:
                opts = {'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-13, 'maxls': 100, 'm': 100}
            else:
                opts = {'maxiter': 20000, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 100, 'm': 100}
            
            res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True, options=opts)
            if hasattr(res, 'x'):
                new_points = _project(res.x.reshape(-1, 3))
                e_new, _ = _energy_grad_vec(new_points)
                e_old, _ = _energy_grad_vec(current)
                if e_new < e_old:
                    current = new_points
        except Exception:
            pass
    
    return current


def _fibonacci_spiral(n):
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = math.pi * (1.0 + math.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project(points)


def _random_points(n, seed):
    """Generate random points on sphere with given seed."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    return _project(points)


def construct():
    global DEADLINE
    DEADLINE = time.time() + 145.0
    
    # Phase 1: Riemannian flow from Fibonacci spiral (primary initialization)
    points_fib = _fibonacci_spiral(N)
    best_points = points_fib.copy()
    best_energy, _ = _energy_grad_vec(best_points)
    
    if _time_left() > 5.0:
        budget1 = min(_time_left() * 0.35, 35.0)
        refined = _riemannian_flow(points_fib, budget1, initial_lr=0.08)
        e, _ = _energy_grad_vec(refined)
        if e < best_energy:
            best_energy = e
            best_points = refined
    
    # Phase 2: A few targeted random restarts (not exhaustive)
    restart_time_limit = _time_left() * 0.3
    restart_start = time.time()
    
    for seed in range(1, 50):
        if time.time() - restart_start > restart_time_limit:
            break
        if _time_left() < 15.0:
            break
        
        random_points = _random_points(N, seed)
        
        # Quick Riemannian flow with limited budget
        budget_per = min(2.0, _time_left() * 0.05)
        if budget_per < 0.3:
            break
        
        refined = _riemannian_flow(random_points, budget_per, initial_lr=0.1, patience=300)
        e, _ = _energy_grad_vec(refined)
        if e < best_energy - 1e-6:  # Only accept significant improvements
            best_energy = e
            best_points = refined
    
    # Phase 3: Aggressive multi-pass L-BFGS-B polish
    if _time_left() > 5.0:
        polish_budget = min(_time_left() * 0.55, 50.0)
        polished = _polish(best_points, polish_budget)
        e, _ = _energy_grad_vec(polished)
        if e < best_energy:
            best_energy = e
            best_points = polished
    
    # Final validation: ensure no points are too close
    best_points = _project(best_points)
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                noise = np.random.randn(3) * 1e-5
                best_points[i] = _project(best_points[i] + noise)
    
    return best_points
