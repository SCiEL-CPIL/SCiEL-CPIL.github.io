"""Combined program: Riemannian gradient descent with exponential map and adaptive line search.
Uses vectorized energy/gradient computation and multiple restarts from structured and random initializations.
"""
import numpy as np
import time
import math

N = 100
_DEADLINE = 115.0  # safety margin before 120s limit


def _project(X):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def _energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _energy_grad(pts):
    """Compute energy and gradient (projected onto tangent space)."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    # Gradient of -sum log(d_ij) with respect to points
    # d/dx_i [-log(d_ij)] = -(x_i - x_j) / d_ij^2
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # grad_i = -sum_j W_ij * x_i + sum_j W_ij * x_j
    row_sums = W.sum(axis=1, keepdims=True)
    grad = -row_sums * pts + (W @ pts)
    
    # Project onto tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad


def _fibonacci_spiral(n, offset=0.5):
    """Generate well-distributed points using a Fibonacci spiral."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(rng):
    """Generate random points on the sphere."""
    v = rng.standard_normal((N, 3))
    return _project(v)


def _gradient_descent(pts, deadline, max_iter=5000, lr_init=0.01, lr_min=1e-12):
    """
    Adaptive gradient descent with backtracking line search and Riemannian exponential map.
    """
    current = pts.copy()
    best_e = _energy(current)
    best_pts = current.copy()
    lr = lr_init
    c1 = 0.1  # Armijo parameter
    rho = 0.5  # backtracking factor
    no_improve_count = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, g = _energy_grad(current)
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-13:
            break
        
        # Backtracking line search
        step = lr
        found = False
        for bt in range(20):
            # Riemannian exponential map: move along geodesic
            v = -step * g  # negative gradient direction
            v_norms = np.linalg.norm(v, axis=1, keepdims=True)
            v_norms_safe = np.where(v_norms < 1e-12, 1.0, v_norms)
            v_unit = v / v_norms_safe
            cos_t = np.cos(v_norms)
            sin_t = np.sin(v_norms)
            new_pts = current * cos_t + v_unit * sin_t
            new_pts = _project(new_pts)
            
            new_e = _energy(new_pts)
            
            # Armijo condition
            if new_e <= e + c1 * step * g_norm * g_norm:
                found = True
                break
            step *= rho
        
        if not found:
            step = max(lr * rho**20, 1e-15)
            v = -step * g
            v_norms = np.linalg.norm(v, axis=1, keepdims=True)
            v_norms_safe = np.where(v_norms < 1e-12, 1.0, v_norms)
            v_unit = v / v_norms_safe
            cos_t = np.cos(v_norms)
            sin_t = np.sin(v_norms)
            new_pts = current * cos_t + v_unit * sin_t
            new_pts = _project(new_pts)
            new_e = _energy(new_pts)
        
        if new_e < best_e - 1e-15:
            current = new_pts
            best_e = new_e
            best_pts = current.copy()
            no_improve_count = 0
            lr = min(step * 1.5, 0.1)
        elif new_e < best_e:
            current = new_pts
            no_improve_count += 1
            lr = step
        else:
            lr = step * 0.5
            no_improve_count += 1
            if lr < lr_min or no_improve_count > 50:
                break
    
    return best_pts, best_e


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Initial guess: Fibonacci spiral
    pts = _fibonacci_spiral(N)
    best_pts = pts.copy()
    best_energy = _energy(pts)
    
    rng = np.random.default_rng(42)
    
    # Optimize the initial guess
    optimized, energy = _gradient_descent(pts.copy(), deadline, max_iter=5000, lr_init=0.01)
    if energy < best_energy:
        best_energy = energy
        best_pts = optimized
    
    # Try random restarts
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        random_pts = _random_sphere(rng)
        optimized, energy = _gradient_descent(random_pts, deadline, max_iter=3000, lr_init=0.01)
        if energy < best_energy:
            best_energy = energy
            best_pts = optimized
    
    # Final polish with smaller step size
    if time.time() < deadline - 5:
        polished, energy = _gradient_descent(best_pts.copy(), deadline, max_iter=5000, lr_init=0.001)
        if energy < best_energy:
            best_energy = energy
            best_pts = polished
    
    return _project(best_pts)
