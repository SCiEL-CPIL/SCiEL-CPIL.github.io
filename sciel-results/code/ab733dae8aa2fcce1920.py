"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def fibonacci_spiral(n):
    """Generate points on unit sphere using golden angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def energy_and_grad(points):
    """
    Compute logarithmic energy and gradient for points on unit sphere.
    Energy = sum_{i<j} log(1/|x_i - x_j|)
    Gradient: dE/dx_i = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(points)
    # Compute pairwise differences and squared distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero on diagonal
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    iu = np.triu_indices(n, k=1)
    dist = np.sqrt(dist_sq[iu])
    energy = -np.sum(np.log(dist))
    
    # Gradient computation
    # grad_i = -sum_{j!=i} (x_i - x_j) / dist_sq[i,j]
    inv_dist_sq = 1.0 / dist_sq  # (n, n)
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # Vectorized sum: grad[i] = -sum_j inv_dist_sq[i,j] * diff[i,j]
    # diff is (n, n, 3), inv_dist_sq is (n, n)
    # Multiply diff by inv_dist_sq expanded to (n, n, 1)
    weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
    grad = -np.sum(weighted_diff, axis=1)
    
    return energy, grad

def tangent_project(points, grad):
    """Project gradient onto tangent space of sphere at each point."""
    dot = np.sum(grad * points, axis=1, keepdims=True)  # (n, 1)
    proj_grad = grad - dot * points  # (n, 3)
    return proj_grad

def optimize_from_start(points, max_iter=2000, gtol=1e-10):
    """
    Optimize points on sphere using L-BFGS-B.
    """
    n = len(points)
    x0 = points.flatten()
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(n, 3)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        x = x / norms
        
        e, g = energy_and_grad(x)
        pg = tangent_project(x, g)
        return e, pg.flatten()
    
    try:
        result = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': gtol, 'ftol': 1e-15}
        )
    except Exception:
        return points
    
    x_opt = result.x.reshape(n, 3)
    norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    x_opt = x_opt / norms
    return x_opt

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Initial best
    best_points = fibonacci_spiral(N)
    best_energy, _ = energy_and_grad(best_points)
    
    # Strategy:
    # 1. Optimize Fibonacci spiral with high precision.
    # 2. Try a few perturbed versions and a random start if time allows.
    # 3. Final high-precision polish on the best.
    
    # Start 1: Pure Fibonacci
    if time.time() < deadline:
        opt_pts = optimize_from_start(best_points, max_iter=3000, gtol=1e-12)
        e, _ = energy_and_grad(opt_pts)
        if e < best_energy:
            best_energy = e
            best_points = opt_pts
    
    # Start 2: Perturbed Fibonacci (small noise)
    if time.time() < deadline:
        rng = np.random.RandomState(123)
        p = best_points.copy()
        noise = rng.randn(N, 3) * 0.01
        p = p + noise
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        p = p / norms
        
        remaining_time = deadline - time.time()
        max_iter = int(remaining_time * 10) 
        if max_iter > 500:
             max_iter = 2000
        if max_iter > 100:
            opt_pts = optimize_from_start(p, max_iter=max_iter, gtol=1e-10)
            e, _ = energy_and_grad(opt_pts)
            if e < best_energy:
                best_energy = e
                best_points = opt_pts

    # Start 3: Another perturbation (larger noise)
    if time.time() < deadline:
        rng = np.random.RandomState(456)
        p = best_points.copy()
        noise = rng.randn(N, 3) * 0.05
        p = p + noise
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        p = p / norms
        
        remaining_time = deadline - time.time()
        max_iter = int(remaining_time * 10)
        if max_iter > 500:
             max_iter = 2000
        if max_iter > 100:
            opt_pts = optimize_from_start(p, max_iter=max_iter, gtol=1e-10)
            e, _ = energy_and_grad(opt_pts)
            if e < best_energy:
                best_energy = e
                best_points = opt_pts
                
    # Start 4: Random Uniform (seeded)
    if time.time() < deadline:
        rng = np.random.RandomState(789)
        p = rng.randn(N, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        p = p / norms
        
        remaining_time = deadline - time.time()
        max_iter = int(remaining_time * 10)
        if max_iter > 500:
             max_iter = 2000
        if max_iter > 100:
            opt_pts = optimize_from_start(p, max_iter=max_iter, gtol=1e-10)
            e, _ = energy_and_grad(opt_pts)
            if e < best_energy:
                best_energy = e
                best_points = opt_pts

    # Final High-Precision Polish
    if time.time() < deadline - 5:
        remaining_time = deadline - time.time()
        # Use very tight tolerance and maximize iterations
        max_iter = int(remaining_time * 20)
        if max_iter > 100:
            # Use a very tight gtol to ensure convergence
            opt_pts = optimize_from_start(best_points, max_iter=max_iter, gtol=1e-15)
            e, _ = energy_and_grad(opt_pts)
            if e < best_energy:
                best_energy = e
                best_points = opt_pts

    # Ensure output is valid
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / np.maximum(norms, 1e-12)
    
    # Verify no points too close
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                best_points[j] += 1e-8 * best_points[i]
                best_points[j] = best_points[j] / np.linalg.norm(best_points[j])
    
    return best_points
# EVOLVE-BLOCK-END
