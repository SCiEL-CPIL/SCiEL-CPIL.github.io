"""Optimized logarithmic energy for 150 points on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE = 150.0  # seconds, evaluator kills at 150s wall-clock
SAFETY_MARGIN = 5.0  # return best before this margin


def _energy_and_grad(points):
    """Compute logarithmic energy and gradient using Gram matrix (O(N^2) matmul).
    
    Energy: sum_{i<j} log(1 / d_ij) where d_ij is chordal distance.
    log(1/d) = -log(d). For points on unit sphere, d_ij^2 = 2 - 2*x_i.x_j.
    """
    # Gram matrix G = X @ X.T, G[i,j] = x_i . x_j
    G = points @ points.T
    # d^2 = 2 - 2*G, clip to avoid log(0)
    d2 = 2.0 - 2.0 * G
    np.fill_diagonal(d2, 1.0)  # diagonal doesn't matter
    d2 = np.maximum(d2, 1e-18)
    
    # Energy: sum_{i<j} -log(d_ij) = -sum_{i<j} 0.5*log(d2_ij)
    # Use upper triangle only
    iu = np.triu_indices(N, 1)
    log_d2 = np.log(d2[iu])
    energy = -0.5 * np.sum(log_d2)
    
    # Gradient: dE/dx_i = sum_{j != i} (x_i - x_j) / d_ij^2
    # From E = -0.5 * sum_{i<j} log(2 - 2 x_i.x_j)
    # dE/dx_i = -0.5 * sum_{j!=i} (-2 * (x_j - x_i)) / (2 - 2 x_i.x_j)
    #          = sum_{j!=i} (x_j - x_i) / (2 - 2 x_i.x_j)
    #          = sum_{j!=i} (x_j - x_i) / d2_ij
    # Note: (x_j - x_i)/d2_ij = -(x_i - x_j)/d2_ij
    # So grad_i = -sum_{j!=i} (x_i - x_j) / d2_ij
    
    # Compute gradient efficiently:
    # grad_i = sum_j (x_j - x_i) / d2_ij for j != i
    # = sum_j x_j / d2_ij - x_i * sum_j 1/d2_ij  (excluding j=i)
    
    inv_d2 = np.zeros((N, N))
    inv_d2[iu] = 1.0 / d2[iu]
    inv_d2[(iu[1], iu[0])] = inv_d2[iu]  # symmetric
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad = X.T @ inv_d2 - X * (inv_d2 @ ones)
    # Actually: grad_i = sum_j inv_d2[i,j] * (x_j - x_i)
    #           = (X.T @ inv_d2)_i - x_i * sum_j inv_d2[i,j]
    row_sums = inv_d2.sum(axis=1)
    grad = points.T @ (inv_d2 @ np.ones(N)) - points * row_sums[:, np.newaxis]
    
    return energy, grad


def _project_to_sphere(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _fibonacci_sphere(n):
    """Golden-angle Fibonacci spiral points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Random points on unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _lbfgsb_optimize(points, max_iter=2000, deadline=None):
    """Run L-BFGS-B on the sphere using gradient projected to tangent space."""
    best_energy, _ = _energy_and_grad(points)
    best_points = points.copy()
    
    def objective_and_grad(x):
        nonlocal best_energy, best_points
        if deadline is not None and time.time() > deadline:
            return best_energy, np.zeros_like(x)
        pts = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(pts)
        # Project gradient to tangent space: g_tan = g - (g.x)x
        g_tan = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        return e, g_tan.flatten()
    
    try:
        result = minimize(
            objective_and_grad,
            points.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': 1e-10, 'maxls': 50}
        )
        if result.success or result.fun < best_energy:
            new_pts = _project_to_sphere(result.x.reshape(N, 3))
            e_new, _ = _energy_and_grad(new_pts)
            if e_new < best_energy:
                best_energy = e_new
                best_points = new_pts
    except Exception:
        pass
    
    return best_points, best_energy


def _riemannian_gd_polish(points, energy, n_steps=5000, deadline=None):
    """Riemannian gradient descent with backtracking line search for final polish."""
    best_points = points.copy()
    best_energy = energy
    
    lr = 0.01
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        _, grad = _energy_and_grad(best_points)
        # Project to tangent space
        grad_tan = grad - np.sum(grad * best_points, axis=1, keepdims=True) * best_points
        
        # Backtracking line search
        alpha = lr
        for _ in range(30):
            # Riemannian exponential map: move along tangent, then renormalize
            new_pts = best_points + alpha * grad_tan
            new_pts = _project_to_sphere(new_pts)
            
            # Check minimum distance
            G = new_pts @ new_pts.T
            d2 = 2.0 - 2.0 * G
            np.fill_diagonal(d2, 1.0)
            min_d2 = d2.min()
            if min_d2 < 1e-18:
                alpha *= 0.5
                continue
            
            e_new, _ = _energy_and_grad(new_pts)
            if e_new < best_energy:
                best_points = new_pts
                best_energy = e_new
                break
            alpha *= 0.5
        
        # Adaptive learning rate
        if alpha > lr * 0.5:
            lr = min(lr * 1.1, 0.05)
        elif alpha < lr * 0.2:
            lr = max(lr * 0.8, 1e-6)
    
    return best_points, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    # Start from Fibonacci spiral
    points = _fibonacci_sphere(N)
    best_points, best_energy = _lbfgsb_optimize(points, max_iter=3000, deadline=deadline)
    
    # Try multiple random restarts if time permits
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        rng_seed = 42 + r * 1000
        random_pts = _random_sphere(N, seed=rng_seed)
        
        # Quick L-BFGS from random start
        pts_r, e_r = _lbfgsb_optimize(random_pts, max_iter=1000, deadline=deadline)
        
        if e_r < best_energy:
            best_energy = e_r
            best_points = pts_r
    
    # Final Riemannian GD polish with tight tolerance
    if time.time() < deadline - 2.0:
        best_points, best_energy = _riemannian_gd_polish(
            best_points, best_energy, n_steps=10000, deadline=deadline
        )
    
    # Ensure valid output
    best_points = _project_to_sphere(best_points)
    # Verify no points too close
    G = best_points @ best_points.T
    d2 = 2.0 - 2.0 * G
    np.fill_diagonal(d2, 1.0)
    min_d = math.sqrt(max(d2.min(), 1e-18))
    if min_d < 1e-9:
        # Fall back to Fibonacci spiral
        best_points = _fibonacci_sphere(N)
    
    return best_points
