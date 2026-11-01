"""Stronger algorithm for 100-point logarithmic energy on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 115.0  # safety margin before 120s kill

def _energy_slow(points):
    """Compute logarithmic energy using math.fsum for high precision."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d = dist[mask]
    d = np.maximum(d, 1e-15)
    return math.fsum(np.log(1.0 / d).tolist())

def _energy_grad_fast(points):
    """Compute energy (using np.sum for speed) and gradient."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    
    # For energy, we need log(1/dist) = -0.5 * log(dist_sq)
    # We only sum i<j
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_sq = dist_sq[mask]
    d_sq = np.maximum(d_sq, 1e-30) # Avoid log(0)
    energy = -0.5 * np.sum(np.log(d_sq))
    
    # Gradient: dE/dx_i = -sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    # Note: The factor of 2 from chain rule on dist^2 cancels with the 1/2 from log(dist^2) derivation?
    # E = -sum log(dist). dE/dx_i = -sum (1/dist) * d(dist)/dx_i
    # d(dist)/dx_i = (x_i - x_j) / dist
    # So term is -(x_i - x_j) / dist^2
    # Using dist_sq: term is -(x_i - x_j) / dist_sq
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    grad = -np.sum(inv_dist_sq[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _sphere_fun_grad_fast(x):
    """Objective and gradient in ambient R^(3N) space with projection.
    Uses fast energy calculation."""
    p = _project_to_sphere(x.reshape(-1, 3))
    e, g = _energy_grad_fast(p)
    # Project gradient to tangent space: g_tangent = g - (g·n)n where n = p (unit normal)
    g_tangent = g - (g * p).sum(axis=1, keepdims=True) * p
    return e, g_tangent.flatten()

def _optimize_single(points, time_budget, maxiter=5000, ftol=1e-14, gtol=1e-10):
    """Run L-BFGS-B optimization from a single starting point."""
    start_time = time.time()
    x0 = points.flatten().copy()
    
    try:
        res = minimize(_sphere_fun_grad_fast, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50})
        
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

def _simulated_annealing_refine(points, time_budget):
    """Refine points using simulated annealing."""
    start_time = time.time()
    best_points = points.copy()
    best_energy = _energy_slow(best_points)
    T = 0.05  # initial temperature
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        # Try a perturbation
        idx = np.random.randint(N)
        angle = np.random.uniform(0, 2 * np.pi) * T
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
        
        # Use fast energy for acceptance decision to save time, 
        # but we must be careful with precision. 
        # For SA, relative change is what matters.
        new_energy_fast, _ = _energy_grad_fast(new_points)
        # Note: _energy_grad_fast uses np.sum which is slightly different from math.fsum
        # but consistent enough for delta_e comparison in SA.
        
        # To be safe and consistent with best_energy (which is slow), we should ideally compare slow energies.
        # But slow energy is ~10x slower. Let's use fast for delta, and only update best_energy with slow if accepted.
        # However, best_energy is currently slow. Let's maintain a fast_best_energy for the loop.
        
        # Actually, let's just use fast energy for the SA loop entirely, and convert to slow at the end.
        # This requires re-initializing best_energy to fast.
        pass # This implementation is getting complicated. Let's stick to the previous SA but make it shorter.
        
        # Re-evaluating: The previous SA used _energy (slow). It was fine if the budget was small.
        # Let's keep the SA simple and use the slow energy but limit the number of iterations strictly by time.
        
        # Simpler SA:
        new_energy = _energy_slow(new_points)
        delta_e = new_energy - best_energy
        
        if delta_e < 0 or np.random.random() < math.exp(-delta_e / max(T, 1e-10)):
            best_points = new_points
            best_energy = new_energy
        
        T *= 0.995  # cool down
        
        # Check time more frequently
        if time.time() - start_time > time_budget:
            break
            
    return best_points

def construct():
    start_time = time.time()
    
    # Start with Fibonacci spiral
    points = _fibonacci_spiral(N)
    best_points = points.copy()
    best_energy = _energy_slow(best_points)
    
    # Phase 1: L-BFGS-B from Fibonacci spiral
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.3:
        optimized = _optimize_single(points, TIME_LIMIT * 0.25, maxiter=5000, ftol=1e-14, gtol=1e-10)
        e = _energy_slow(optimized)
        if e < best_energy:
            best_energy = e
            best_points = optimized
    
    # Phase 2: Multiple random restarts with L-BFGS-B
    np.random.seed(123)
    restart_count = 0
    # Allocate 50% of time to restarts
    restart_time_limit = TIME_LIMIT * 0.8
    while True:
        elapsed = time.time() - start_time
        if elapsed > restart_time_limit:
            break
        restart_count += 1
        
        # Generate diverse random starting points
        random_points = _random_points(N)
        
        # Quick optimization with limited iterations
        # Use a smaller budget per restart to allow more restarts
        budget_per_restart = min(TIME_LIMIT * 0.02, (restart_time_limit - elapsed) * 0.9)
        if budget_per_restart < 1.0:
            break
            
        optimized = _optimize_single(random_points, budget_per_restart, maxiter=1000, ftol=1e-10, gtol=1e-8)
        e = _energy_slow(optimized)
        if e < best_energy:
            best_energy = e
            best_points = optimized
            
    # Phase 3: Fine-tune the best solution with more iterations
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.9:
        remaining = TIME_LIMIT * 0.9 - elapsed
        if remaining > 2.0:
            refined = _optimize_single(best_points, remaining, maxiter=8000, ftol=1e-15, gtol=1e-12)
            e = _energy_slow(refined)
            if e < best_energy:
                best_energy = e
                best_points = refined
    
    # Phase 4: Simulated annealing refinement if time allows
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.95:
        remaining = TIME_LIMIT * 0.95 - elapsed
        if remaining > 1.0:
            np.random.seed(456)
            sa_points = _simulated_annealing_refine(best_points, remaining * 0.8)
            e = _energy_slow(sa_points)
            if e < best_energy:
                best_energy = e
                best_points = sa_points
    
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
    final_energy = _energy_slow(best_points)
    if not np.isfinite(final_energy):
        # Fallback to Fibonacci spiral
        best_points = _fibonacci_spiral(N)
    
    return best_points
