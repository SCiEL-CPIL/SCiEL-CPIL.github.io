"""Stronger algorithm for 100-point logarithmic energy on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 145.0  # safety margin before 150s kill


def _energy_fsum(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dist[mask]
    d = np.maximum(d, 1e-15)
    return math.fsum(np.log(1.0 / d).tolist())


def _energy_grad_vec(points):
    """Compute energy and gradient of logarithmic energy.
    
    E = sum_{i<j} log(1/||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    dE/dx_i = -sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    """
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    grad = -np.einsum('ij,ijk->ik', inv_dist_sq, diff)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dist[mask]
    d = np.maximum(d, 1e-15)
    energy = math.fsum(np.log(1.0 / d).tolist())
    
    return energy, grad


def _project(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _riemannian_flow(points, time_budget, max_iter=20000, ftol=1e-15, gtol=1e-12):
    """Riemannian gradient flow with adaptive step size on the sphere.
    
    Uses exponential map (rotation) for exact sphere constraint satisfaction.
    """
    start_time = time.time()
    p = points.copy()
    e, g = _energy_grad_vec(p)
    best_e = e
    best_p = p.copy()
    
    # Project gradient to tangent space
    g_tan = g - np.einsum('ij,ij->i', g, p)[:, np.newaxis] * p
    
    # Adaptive step size
    step = 0.01
    min_step = 1e-12
    max_step = 0.5
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        g_norm = np.linalg.norm(g_tan)
        if g_norm < gtol:
            break
        
        # Normalize gradient direction
        g_dir = g_tan / (g_norm + 1e-30)
        
        # Take step using exponential map (rotation)
        # For small angles, rotation ≈ addition; use exact rotation for stability
        angle = step * g_norm
        if angle < 1e-10:
            new_p = p + step * g_tan
            new_p = _project(new_p)
        else:
            # Rotation by angle around axis g_dir
            # Use Rodrigues' formula
            axis = g_dir
            c = math.cos(angle)
            s = math.sin(angle)
            k_outer_k = np.einsum('i,j->ij', axis, axis)
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R = c * np.eye(3) + s * K + (1 - c) * k_outer_k
            new_p = p @ R.T
        
        new_e = _energy_fsum(new_p)
        
        if new_e < e - ftol:
            # Accept step
            e = new_e
            p = new_p
            step = min(step * 1.05, max_step)
            
            if e < best_e:
                best_e = e
                best_p = p.copy()
            
            # Recompute gradient
            _, g = _energy_grad_vec(p)
            g_tan = g - np.einsum('ij,ij->i', g, p)[:, np.newaxis] * p
        else:
            # Reject step
            step *= 0.5
            if step < min_step:
                break
    
    return best_p


def _polish_lbfgs(points, time_budget, maxiter=5000, ftol=1e-15, gtol=1e-13):
    """Polish using L-BFGS-B in ambient space with sphere projection."""
    start_time = time.time()
    
    def fun_grad(x):
        p = _project(x.reshape(-1, 3))
        e, g = _energy_grad_vec(p)
        g_tan = g - np.einsum('ij,ij->i', g, p)[:, np.newaxis] * p
        return e, g_tan.flatten()
    
    x0 = points.flatten().copy()
    
    try:
        res = minimize(fun_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50})
        
        if hasattr(res, 'x') and res.x is not None:
            new_points = _project(res.x.reshape(-1, 3))
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
    return _project(points)


def _random_points(n, seed=None):
    """Generate random points on sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    points = rng.randn(n, 3)
    return _project(points)


def _kick_and_optimize(points, time_budget):
    """Perturb points and re-optimize to escape local minima."""
    # Random perturbation
    perturb = np.random.randn(N, 3) * 0.05
    new_points = _project(points + perturb)
    
    # Optimize
    result = _polish_lbfgs(new_points, time_budget * 0.7, maxiter=3000)
    result = _riemannian_flow(result, time_budget * 0.3, max_iter=5000)
    
    return result


def construct():
    start_time = time.time()
    
    # Start with Fibonacci spiral
    points = _fibonacci_spiral(N)
    best_points = points.copy()
    best_energy = _energy_fsum(best_points)
    
    # Phase 1: Riemannian flow from Fibonacci spiral
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.25:
        optimized = _riemannian_flow(points, TIME_LIMIT * 0.2, max_iter=30000)
        e = _energy_fsum(optimized)
        if e < best_energy:
            best_energy = e
            best_points = optimized
    
    # Phase 2: L-BFGS-B polish
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.4:
        refined = _polish_lbfgs(best_points, TIME_LIMIT * 0.15, maxiter=5000)
        e = _energy_fsum(refined)
        if e < best_energy:
            best_energy = e
            best_points = refined
    
    # Phase 3: Multiple diverse restarts
    np.random.seed(42)
    restart_count = 0
    while True:
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT * 0.65:
            break
        restart_count += 1
        
        # Generate diverse starting points
        if restart_count % 4 == 1:
            # Random points
            random_points = _random_points(N)
        elif restart_count % 4 == 2:
            # Fibonacci with random rotation
            base = _fibonacci_spiral(N)
            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            angle = np.random.uniform(0, 2 * np.pi)
            c = math.cos(angle)
            s = math.sin(angle)
            k = axis
            k_outer_k = np.outer(k, k)
            K = np.array([[0, -k[2], k[1]],
                          [k[2], 0, -k[0]],
                          [-k[1], k[0], 0]])
            R = c * np.eye(3) + s * K + (1 - c) * k_outer_k
            random_points = _project(base @ R.T)
        elif restart_count % 4 == 3:
            # Perturbed Fibonacci
            base = _fibonacci_spiral(N)
            perturb = np.random.randn(N, 3) * 0.15
            random_points = _project(base + perturb)
        else:
            # Spherical cap sampling
            random_points = _random_points(N)
        
        # Quick optimization
        budget_per_restart = min(TIME_LIMIT * 0.04, (TIME_LIMIT * 0.65 - elapsed) * 0.85)
        if budget_per_restart < 1.0:
            break
        
        optimized = _polish_lbfgs(random_points, budget_per_restart, maxiter=2000)
        e = _energy_fsum(optimized)
        if e < best_energy:
            best_energy = e
            best_points = optimized
    
    # Phase 4: Kick and optimize to escape local minima
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.8:
        remaining = TIME_LIMIT * 0.8 - elapsed
        if remaining > 3.0:
            kicked = _kick_and_optimize(best_points, remaining * 0.9)
            e = _energy_fsum(kicked)
            if e < best_energy:
                best_energy = e
                best_points = kicked
    
    # Phase 5: Final Riemannian flow polish
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.92:
        remaining = TIME_LIMIT * 0.92 - elapsed
        if remaining > 2.0:
            refined = _riemannian_flow(best_points, remaining * 0.95, max_iter=50000, ftol=1e-16, gtol=1e-14)
            e = _energy_fsum(refined)
            if e < best_energy:
                best_energy = e
                best_points = refined
    
    # Phase 6: Final L-BFGS-B high-precision polish
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.97:
        remaining = TIME_LIMIT * 0.97 - elapsed
        if remaining > 1.0:
            final_opt = _polish_lbfgs(best_points, remaining * 0.95, maxiter=10000, ftol=1e-16, gtol=1e-14)
            e = _energy_fsum(final_opt)
            if e < best_energy:
                best_energy = e
                best_points = final_opt
    
    # Final projection and validation
    best_points = _project(best_points)
    
    # Check for points too close together
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                perturb = np.random.randn(3) * 1e-5
                best_points[i] = best_points[i] + perturb
                best_points[i] = best_points[i] / np.linalg.norm(best_points[i])
    
    # Final energy check
    final_energy = _energy_fsum(best_points)
    if not np.isfinite(final_energy):
        best_points = _fibonacci_spiral(N)
    
    return best_points
