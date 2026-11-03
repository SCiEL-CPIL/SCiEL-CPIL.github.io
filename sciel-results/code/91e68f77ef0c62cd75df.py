"""
Minimize logarithmic energy of 150 points on the unit sphere using 
multiple restarts and projected gradient descent with backtracking line search.
"""
import time
import math
import numpy as np

N = 150
TIME_LIMIT = 115.0  # Safety margin before 120s kill

def _energy_and_grad(pos):
    """
    Compute logarithmic energy and gradient for points on unit sphere.
    
    pos: (N, 3) array of points on unit sphere.
    Returns (energy, grad) where grad is the unconstrained gradient w.r.t. pos.
    Energy is sum_{i<j} log(1 / |x_i - x_j|).
    """
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid self-pairs (diagonal)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    energy = -np.sum(np.log(d))
    
    # Gradient computation:
    # d/dp_i [ -log(|p_i - p_j|) ] = -(p_i - p_j) / |p_i - p_j|^2
    # grad_i = sum_{j != i} (p_j - p_i) / |p_i - p_j|^2
    #         = sum_{j != i} p_j / d_ij^2 - p_i * sum_{j != i} 1/d_ij^2
    
    inv_d2 = 1.0 / dist_sq
    np.fill_diagonal(inv_d2, 0.0) # Exclude self-interaction for gradient
    
    # sum_j p_j / d_ij^2 for each i
    sum_pj_over_d2 = inv_d2 @ pos
    # sum_j 1/d_ij^2 for each i
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    
    grad = sum_pj_over_d2 - pos * sum_inv_d2[:, np.newaxis]
    
    return energy, grad

def _project_to_sphere(pos):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pos, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pos / norms

def _optimize(pos_init, time_budget):
    """
    Optimize a single starting configuration using projected gradient descent 
    with backtracking line search.
    """
    start_time = time.time()
    pos = _project_to_sphere(pos_init.copy())
    energy, grad = _energy_and_grad(pos)
    best_pos = pos.copy()
    best_energy = energy
    
    # Initial learning rate
    lr = 0.01
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        # Project gradient onto tangent space: grad_t = grad - (grad . x) x
        grad_dot_x = np.sum(grad * pos, axis=1, keepdims=True)
        grad_tangent = grad - grad_dot_x * pos
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < 1e-10:
            break
            
        # Backtracking line search
        # Try to take a step in the negative gradient direction
        # We want to minimize energy, so we move in -grad_tangent direction
        direction = -grad_tangent
        
        # Normalize direction to make lr meaningful as a step size on the sphere
        dir_norm = np.linalg.norm(direction, axis=1, keepdims=True)
        dir_norm[dir_norm < 1e-15] = 1.0
        direction = direction / dir_norm
        
        # Start with current lr
        alpha = lr
        current_energy = energy
        current_pos = pos
        
        # Backtracking
        rho = 0.5
        c = 1e-4 # Armijo condition parameter
        
        for _ in range(20): # Max backtracks
            new_pos = current_pos + alpha * direction
            new_pos = _project_to_sphere(new_pos)
            
            new_energy, new_grad = _energy_and_grad(new_pos)
            
            # Armijo condition: f(x + alpha*d) <= f(x) + c * alpha * grad^T d
            # Here d = -grad_tangent, so grad^T d = -||grad_tangent||^2
            # f(new) <= f(old) - c * alpha * ||grad_tangent||^2
            
            if new_energy <= current_energy - c * alpha * (grad_norm**2):
                # Accept step
                pos = new_pos
                energy = new_energy
                grad = new_grad
                break
            else:
                alpha *= rho
        
        else:
            # If no step accepted, reduce lr significantly and try again or break
            lr *= 0.1
            if alpha < 1e-12:
                break
            continue
            
        # If step accepted, increase lr slightly for next iteration (optional)
        # lr = min(lr * 1.1, 0.1) 
        
        # Update best
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
            
    return _project_to_sphere(best_pos), best_energy

def construct():
    start_time = time.time()
    
    # Generate a list of starting configurations
    starts = []
    
    # 1. Fibonacci spiral (standard)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pos_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    starts.append(_project_to_sphere(pos_fib))
    
    # 2. Fibonacci spiral with small perturbation
    rng = np.random.default_rng(42)
    pos_fib_pert = pos_fib + rng.normal(0, 0.01, size=(N, 3))
    starts.append(_project_to_sphere(pos_fib_pert))
    
    # 3. Random uniform points
    for _ in range(3):
        pos_rand = rng.normal(0, 1, size=(N, 3))
        starts.append(_project_to_sphere(pos_rand))
        
    # 4. Maybe a different spiral or pattern? 
    # Let's try a few more random ones
    for _ in range(2):
        pos_rand = rng.normal(0, 1, size=(N, 3))
        starts.append(_project_to_sphere(pos_rand))

    best_pos = starts[0]
    best_energy = float('inf')
    
    # Allocate time budget to each start
    # We have TIME_LIMIT seconds. 
    # Let's give 60% of time to the first few most promising starts (Fibonacci)
    # and the rest to random ones.
    
    # Sort starts? No, keep order. 
    # Time per start:
    # Total time available for optimization: TIME_LIMIT - 5 (buffer)
    # Number of starts: 7
    # Let's give 30% to the first one, 20% to the second, and the rest split.
    
    total_opt_time = TIME_LIMIT - 10.0 # 10s buffer for overhead and final checks
    time_per_start = [
        total_opt_time * 0.30, # Fib
        total_opt_time * 0.20, # Fib Pert
        total_opt_time * 0.10, # Rand 1
        total_opt_time * 0.10, # Rand 2
        total_opt_time * 0.10, # Rand 3
        total_opt_time * 0.10, # Rand 4
        total_opt_time * 0.10, # Rand 5
    ]
    
    for idx, start_pos in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 5:
            break
            
        # Ensure we don't exceed total limit
        alloc_time = min(time_per_start[idx], TIME_LIMIT - 5 - elapsed)
        if alloc_time < 1.0:
            continue
            
        pos, energy = _optimize(start_pos, alloc_time)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
            
    # Final validation and projection
    best_pos = _project_to_sphere(best_pos)
    
    if not np.all(np.isfinite(best_pos)):
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        best_pos = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        best_pos = _project_to_sphere(best_pos)
        
    # Check minimum distance
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # Perturb and retry quickly
        rng = np.random.default_rng(123)
        perturbed = best_pos + rng.normal(0, 0.01, size=(N, 3))
        perturbed = _project_to_sphere(perturbed)
        # Quick optimization
        pos, energy = _optimize(perturbed, 5.0)
        if energy < best_energy:
            best_pos = pos
            best_energy = energy
        else:
            best_pos = perturbed # Just to ensure they are distinct, though energy might be worse
            # Actually, if min_dist is too small, the energy is likely huge (log(1/0) -> inf).
            # So the perturbed one is better.
            
    return best_pos
