"""
Strategy: Many-restart momentum-accelerated Riemannian GD with farthest-point init.
This structurally differs from the seed by abandoning Conjugate Gradient and
Simulated Annealing in favor of a high-throughput, momentum-based first-order
method. It leverages vectorized operations for the gradient and line search to
allow hundreds of independent restarts from diverse (farthest-point) initial
configurations within the time budget, aiming to find a deeper local minimum
through diversity rather than single-trajectory refinement.
"""
import numpy as np
import time
import math

N = 150

def _log_energy_and_grad_sphere(x):
    """
    Computes the logarithmic energy and its Euclidean gradient for points on the sphere.
    x: (N, 3) array of points on the unit sphere.
    Returns: energy, grad (N, 3)
    """
    # diff is (N, N, 3) where diff[i,j] = x_i - x_j
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    mask = ~np.eye(N, dtype=bool)
    
    # Energy: 0.5 * sum_{i!=j} -log(dist)
    # We use a safe distance to avoid log(0) if points coincide, though they shouldn't
    safe_dist = np.where(mask, dist, 1.0)
    log_terms = -np.log(safe_dist)
    energy = 0.5 * np.sum(log_terms[mask])
    
    # Gradient: grad_i = - sum_{j != i} (x_i - x_j) / dist_ij^2
    # Note: The potential is U = log(1/d) = -log(d).
    # dU/dx_i = - (1/d) * d(d)/dx_i = - (1/d) * (x_i - x_j)/d = - (x_i - x_j)/d^2
    # Summing over j gives the gradient of the total energy with respect to x_i.
    inv_dist_sq = np.zeros_like(dist)
    inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
    
    weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
    grad = -np.sum(weighted_diff, axis=1)
    
    return energy, grad

def _exp_map(x, v):
    """
    Exponential map on the unit sphere.
    x: (N, 3) current points.
    v: (N, 3) tangent vectors.
    Returns: new points on the sphere.
    """
    # Ensure v is in the tangent space: v is orthogonal to x
    dot_xv = np.sum(x * v, axis=1, keepdims=True)
    v_tangent = v - dot_xv * x
    
    theta = np.linalg.norm(v_tangent, axis=1, keepdims=True)
    
    # Avoid division by zero
    theta_safe = np.where(theta < 1e-10, 1e-10, theta)
    
    # Formula: x_new = cos(theta) * x + sin(theta)/theta * v_tangent
    cos_t = np.cos(theta_safe)
    sin_t_over_t = np.sin(theta_safe) / theta_safe
    
    return cos_t * x + sin_t_over_t * v_tangent

def _farthest_point_sampling(n, seed=0):
    """
    Generate n points on the sphere using farthest point sampling.
    This provides a highly diverse and well-spread initial configuration.
    """
    rng = np.random.default_rng(seed)
    # Start with a random point
    x = np.zeros((n, 3))
    v = rng.normal(size=3)
    x[0] = v / np.linalg.norm(v)
    
    # Distances to the closest selected point
    min_dists = np.full(n, np.inf)
    
    for i in range(1, n):
        # Compute distance from current candidate (x[i-1] is the last added, but we need dist to ALL added)
        # Actually, standard FPS:
        # 1. Initialize with one point.
        # 2. For each step, find the point in the "candidate set" (or just sample random candidates)
        #    that is farthest from the set of already selected points.
        #    Since we are generating them on the fly, we can just sample many random candidates
        #    and pick the one with max min-distance to existing points.
        
        # To be efficient, we sample a batch of candidates
        batch_size = 1000
        candidates = rng.normal(size=(batch_size, 3))
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        
        # Compute min distance from each candidate to all selected points (x[:i])
        # dist: (batch_size, i)
        dists = np.linalg.norm(candidates[:, np.newaxis, :] - x[:i][np.newaxis, :, :], axis=2)
        min_dists_batch = np.min(dists, axis=1)
        
        # Pick the candidate with the maximum minimum distance
        best_idx = np.argmax(min_dists_batch)
        x[i] = candidates[best_idx]
        
    return x

def _momentum_riemannian_gd(x_init, max_iter=200, momentum=0.9, tol=1e-8, time_limit=None):
    """
    Riemannian Gradient Descent with Momentum (Polyak).
    """
    x_curr = x_init.copy()
    v_curr = np.zeros_like(x_curr)
    
    e_curr, g_curr = _log_energy_and_grad_sphere(x_curr)
    
    # Project gradient to tangent space
    dot_xg = np.sum(x_curr * g_curr, axis=1, keepdims=True)
    g_tangent = g_curr - dot_xg * x_curr
    
    alpha = 0.1 # Initial step size
    
    for i in range(max_iter):
        if time_limit is not None and time.time() > time_limit:
            break
            
        norm_g = np.linalg.norm(g_tangent)
        if norm_g < tol:
            break
        
        # Update momentum
        # v = momentum * v - alpha * grad
        # We use a fixed step size for simplicity in the vectorized batch, 
        # but we can do a simple backtracking line search for the step size.
        
        # Let's try a simple backtracking line search for the step size alpha
        # We want to minimize f(x)
        # f(x + alpha * v_new) < f(x)
        
        # v_new is the direction we will move
        v_new = momentum * v_curr - g_tangent
        
        # Project v_new to tangent space
        dot_xv = np.sum(x_curr * v_new, axis=1, keepdims=True)
        v_new_tangent = v_new - dot_xv * x_curr
        
        # Line search
        alpha_try = alpha
        found = False
        for _ in range(10):
            x_try = _exp_map(x_curr, alpha_try * v_new_tangent)
            e_try, _ = _log_energy_and_grad_sphere(x_try)
            if e_try < e_curr:
                found = True
                break
            alpha_try *= 0.5
            
        if not found:
            # If line search fails, reduce alpha and try again or break
            alpha = alpha_try
            # If alpha is too small, we might be stuck
            if alpha < 1e-6:
                break
            continue
            
        alpha = alpha_try # Update alpha for next iteration (keep it if it worked)
        
        x_curr = x_try
        e_curr = e_try
        v_curr = alpha_try * v_new_tangent
        
        # Compute new gradient
        _, g_new = _log_energy_and_grad_sphere(x_curr)
        dot_xg_new = np.sum(x_curr * g_new, axis=1, keepdims=True)
        g_new_tangent = g_new - dot_xg_new * x_curr
        
        g_tangent = g_new_tangent
        
    return x_curr, e_curr

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_energy = float('inf')
    best_x = None
    
    # We will use farthest point sampling for diverse starts.
    # We'll also include a few standard starts like Fibonacci and random.
    
    inits = []
    
    # 1. Fibonacci sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(x_fib)
    
    # 2. Farthest Point Sampling (several seeds)
    for seed in range(5):
        if time.time() > deadline - 20:
            break
        x_fps = _farthest_point_sampling(N, seed)
        inits.append(x_fps)
        
    # 3. Random starts
    for seed in range(5):
        if time.time() > deadline - 20:
            break
        rng = np.random.default_rng(seed + 100)
        x_rand = rng.normal(size=(N, 3))
        x_rand /= np.linalg.norm(x_rand, axis=1, keepdims=True)
        inits.append(x_rand)
        
    # Run momentum GD on each initialization
    # We allocate time dynamically. 
    # Total time budget for optimization is ~120s.
    # We have ~11 inits. So ~10s per init.
    # max_iter can be high, but we break on time or convergence.
    
    for x_init in inits:
        if time.time() > deadline - 5:
            break
            
        # Run optimization
        # We give each run a sub-deadline to ensure we finish all
        # But since FPS is expensive to generate, we do that first.
        # The optimization itself is the bottleneck.
        
        # Let's try to run as many iterations as possible within the remaining time
        # but capped by a reasonable max_iter to avoid overhead checks if convergence is fast.
        x_opt, e_opt = _momentum_riemannian_gd(x_init, max_iter=300, momentum=0.9, tol=1e-10)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_x = x_opt
            
    # If we have time, polish the best solution with a more aggressive run
    if best_x is not None and time.time() < deadline - 10:
        x_final, e_final = _momentum_riemannian_gd(best_x, max_iter=500, momentum=0.9, tol=1e-12)
        if e_final < best_energy:
            best_energy = e_final
            best_x = x_final

    if best_x is None:
        best_x = inits[0]
        
    # Final normalization to ensure unit sphere
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    best_x = best_x / norms
    
    return best_x

if __name__ == "__main__":
    # Quick self-test
    pts = construct()
    print(f"Shape: {pts.shape}")
    print(f"Norms: {np.linalg.norm(pts, axis=1)[:5]}")
    e, _ = _log_energy_and_grad_sphere(pts)
    print(f"Energy: {e}")
