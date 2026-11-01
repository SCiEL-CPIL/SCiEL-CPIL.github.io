"""Stronger algorithm for 100-point logarithmic energy on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 145.0  # safety margin before 150s kill


def _energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_sq = dist_sq[mask]
    d = np.sqrt(np.maximum(d_sq, 1e-30))
    return math.fsum(np.log(1.0 / d).tolist())


def _energy_grad(points):
    """Compute energy and gradient of logarithmic energy on the sphere.
    
    E = sum_{i<j} log(1/||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    dE/dx_i = -sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    """
    n = len(points)
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero on diagonal
    
    # inv_dist_sq[i,j] = 1 / dist[i,j]^2
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = -sum_j (points[i] - points[j]) / dist[i,j]^2
    grad = -np.sum(inv_dist_sq[:, :, np.newaxis] * diff, axis=1)
    
    # Compute energy
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_sq = dist_sq[mask]
    d = np.sqrt(np.maximum(d_sq, 1e-30))
    energy = math.fsum(np.log(1.0 / d).tolist())
    
    return energy, grad


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _sphere_fun_grad(x):
    """Objective and gradient in ambient R^(3N) space with projection."""
    p = _project_to_sphere(x.reshape(-1, 3))
    e, g = _energy_grad(p)
    # Project gradient to tangent space: g_tangent = g - (g·n)n where n = p (unit normal)
    g_tangent = g - (g * p).sum(axis=1, keepdims=True) * p
    return e, g_tangent.flatten()


def _optimize_single(points, time_budget, maxiter=3000):
    """Run L-BFGS-B optimization from a single starting point."""
    start_time = time.time()
    x0 = points.flatten().copy()
    
    try:
        res = minimize(_sphere_fun_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50})
        
        if hasattr(res, 'x'):
            new_points = _project_to_sphere(res.x.reshape(-1, 3))
            return new_points
    except Exception:
        pass
    
    return points


def _fibonacci_spiral(n):
    """Generate initial points using Fibonacci spiral on sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(points)


def _random_points(n, seed=None):
    """Generate random points on sphere."""
    if seed is not None:
        np.random.seed(seed)
    points = np.random.randn(n, 3)
    return _project_to_sphere(points)


def _steepest_descent_sphere(points, time_budget, max_iter=100000):
    """Steepest descent on the sphere with adaptive step size."""
    start_time = time.time()
    p = points.copy()
    best_energy = _energy(p)
    best_points = p.copy()
    
    step = 0.01
    min_step = 1e-10
    max_step = 0.1
    
    for _ in range(max_iter):
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        e, grad = _energy_grad(p)
        
        # Project gradient to tangent space
        grad_tangent = grad - (grad * p).sum(axis=1, keepdims=True) * p
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < 1e-12:
            break
        
        # Normalize gradient direction
        direction = -grad_tangent / grad_norm
        
        # Adaptive step size (backtracking line search)
        accepted = False
        s = step
        while s > min_step:
            new_p = p + s * direction
            new_p = _project_to_sphere(new_p)
            new_e = _energy(new_p)
            
            if new_e < e - 1e-12 * s * grad_norm:
                p = new_p
                step = min(s * 1.5, max_step)
                accepted = True
                break
            s *= 0.5
        
        if not accepted:
            step = max(s * 0.5, min_step)
        
        e = _energy(p)
        if e < best_energy:
            best_energy = e
            best_points = p.copy()
    
    return best_points


def _simulated_annealing_refine(points, time_budget):
    """Refine points using simulated annealing with efficient energy updates."""
    start_time = time.time()
    best_points = points.copy()
    best_energy = _energy(best_points)
    T = 0.05  # initial temperature
    cooling_rate = 0.998
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        # Try a perturbation: move one point
        idx = np.random.randint(N)
        angle = np.random.uniform(0, 2 * np.pi) * T * 2
        # Rotate point idx by small angle around a random axis
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        
        # Rodrigues rotation
        c = math.cos(angle)
        s = math.sin(angle)
        k = axis
        k_outer_k = np.outer(k, k)
        K = np.array([[0, -k[2], k[1]],
                      [k[2], 0, -k[0]],
                      [-k[1], k[0], 0]])
        R = c * np.eye(3) + s * K + (1 - c) * k_outer_k
        
        new_points = best_points.copy()
        new_points[idx] = R @ new_points[idx]
        new_points = _project_to_sphere(new_points)
        
        new_energy = _energy(new_points)
        delta_e = new_energy - best_energy
        
        if delta_e < 0 or np.random.random() < math.exp(-delta_e / max(T, 1e-10)):
            best_points = new_points
            best_energy = new_energy
        
        T *= cooling_rate
        if T < 1e-8:
            break
    
    return best_points


def construct():
    start_time = time.time()
    
    # Start with Fibonacci spiral
    points = _fibonacci_spiral(N)
    best_points = points.copy()
    best_energy = _energy(best_points)
    
    # Phase 1: L-BFGS-B from Fibonacci spiral
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.3:
        optimized = _optimize_single(points, TIME_LIMIT * 0.25, maxiter=5000)
        e = _energy(optimized)
        if e < best_energy:
            best_energy = e
            best_points = optimized
    
    # Phase 2: Multiple random restarts with L-BFGS-B
    np.random.seed(123)
    restart_count = 0
    while True:
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT * 0.65:
            break
        restart_count += 1
        
        # Generate diverse random starting points
        random_points = _random_points(N)
        
        # Quick optimization with limited iterations
        budget_per_restart = min(TIME_LIMIT * 0.025, (TIME_LIMIT * 0.65 - elapsed) * 0.8)
        if budget_per_restart < 1.0:
            break
            
        optimized = _optimize_single(random_points, budget_per_restart, maxiter=1000)
        e = _energy(optimized)
        if e < best_energy:
            best_energy = e
            best_points = optimized
    
    # Phase 3: Steepest descent polish on best solution
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.85:
        remaining = TIME_LIMIT * 0.85 - elapsed
        if remaining > 2.0:
            refined = _steepest_descent_sphere(best_points, remaining, max_iter=50000)
            e = _energy(refined)
            if e < best_energy:
                best_energy = e
                best_points = refined
    
    # Phase 4: Simulated annealing refinement if time allows
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.95:
        remaining = TIME_LIMIT * 0.95 - elapsed
        if remaining > 1.0:
            np.random.seed(456)
            sa_points = _simulated_annealing_refine(best_points, remaining * 0.7)
            e = _energy(sa_points)
            if e < best_energy:
                best_energy = e
                best_points = sa_points
    
    # Phase 5: Final L-BFGS polish with tight tolerance
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.98:
        remaining = TIME_LIMIT * 0.98 - elapsed
        if remaining > 1.0:
            final = _optimize_single(best_points, remaining, maxiter=3000)
            e = _energy(final)
            if e < best_energy:
                best_energy = e
                best_points = final
    
    # Final projection and validation
    best_points = _project_to_sphere(best_points)
    
    # Check for points too close together
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                # Perturb point i
                perturb = np.random.randn(3) * 1e-5
                best_points[i] = best_points[i] + perturb
                best_points[i] = best_points[i] / np.linalg.norm(best_points[i])
    
    # Final energy check
    final_energy = _energy(best_points)
    if not np.isfinite(final_energy):
        # Fallback to Fibonacci spiral
        best_points = _fibonacci_spiral(N)
    
    return best_points
