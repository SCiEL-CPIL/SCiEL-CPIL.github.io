"""
Strategy: Deterministic multi-restart projected gradient descent with analytic gradients.
We use the analytic gradient of the logarithmic energy to perform fast, precise local optimization.
Multiple diverse starting points (Fibonacci sphere, random, and perturbed) are used to explore
different basins of attraction. Adaptive step size control (barrier method style) ensures
robust convergence without getting stuck due to step size issues.
"""
import numpy as np
import time
import math

N = 150
DEADLINE = 145.0  # Safety margin for the 150s limit

def _energy_and_grad(points):
    """
    Compute logarithmic energy and its gradient.
    Energy E = sum_{i<j} log(1 / |x_i - x_j|)
    Gradient w.r.t x_k: sum_{j != k} (x_k - x_j) / |x_k - x_j|^2
    """
    # Compute pairwise distances
    # dist_sq[i, j] = |x_i - x_j|^2
    # G = points @ points.T
    # dist_sq = 2 - 2 * G
    G = points @ points.T
    dist_sq = 2.0 - 2.0 * G
    np.fill_diagonal(dist_sq, 1.0) # Avoid division by zero, though we mask it later
    
    # Energy calculation
    # We need sum over i < j of log(1/sqrt(dist_sq)) = -0.5 * sum_{i<j} log(dist_sq)
    iu = np.triu_indices(N, k=1)
    vals = dist_sq[iu]
    # Clip for log safety
    vals = np.maximum(vals, 1e-15)
    energy = -0.5 * np.sum(np.log(vals))
    
    # Gradient calculation
    # grad_k = sum_{j != k} (x_k - x_j) / dist_sq[k, j]
    # Let D be the matrix of 1/dist_sq
    D = 1.0 / dist_sq
    np.fill_diagonal(D, 0.0)
    
    # grad_k = sum_j D[k, j] * (x_k - x_j)
    # = x_k * sum_j D[k, j] - sum_j D[k, j] * x_j
    # = x_k * row_sum_k - (D @ points)[k]
    
    row_sums = np.sum(D, axis=1)
    D_x = D @ points
    grad = points * row_sums[:, np.newaxis] - D_x
    
    return energy, grad

def _project_to_tangent(points, grad):
    """
    Project gradient onto the tangent space of the sphere at each point.
    grad_tan_k = grad_k - (grad_k . x_k) x_k
    """
    dots = np.sum(grad * points, axis=1)
    return grad - dots[:, np.newaxis] * points

def _optimize_local(points, time_limit=10.0):
    """
    Run projected gradient descent with adaptive step size.
    """
    current = points.copy()
    current_energy, current_grad = _energy_and_grad(current)
    best_points = current.copy()
    best_energy = current_energy
    
    start_time = time.time()
    lr = 0.01 # Initial learning rate
    min_lr = 1e-10
    max_lr = 0.1
    
    # Pre-compute indices for energy check if needed, but we use _energy_and_grad
    
    while time.time() - start_time < time_limit:
        # Project gradient
        grad_tan = _project_to_tangent(current, current_grad)
        
        # Check if gradient is small
        grad_norm = np.linalg.norm(grad_tan)
        if grad_norm < 1e-10:
            break
            
        # Adaptive step size: try to move
        # We want to decrease energy.
        # new_point = current - lr * grad_tan
        # We need to renormalize.
        
        step = lr * grad_tan
        new_points = current - step
        
        # Renormalize
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_points = new_points / norms
        
        new_energy, new_grad = _energy_and_grad(new_points)
        
        if new_energy < current_energy:
            # Accept step
            current = new_points
            current_energy = new_energy
            current_grad = new_grad
            
            if current_energy < best_energy:
                best_points = current.copy()
                best_energy = current_energy
                
            # Increase learning rate slightly if we made good progress
            # lr *= 1.1
        else:
            # Reject step, decrease learning rate
            lr *= 0.5
            if lr < min_lr:
                break
                
    return best_points, best_energy

def construct():
    start_time = time.time()
    
    # Initial best: Fibonacci sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = fib_points
    best_energy, _ = _energy_and_grad(fib_points)
    
    # Generate diverse starting points
    starts = []
    
    # 1. Fibonacci sphere
    starts.append(fib_points)
    
    # 2. Random points
    np.random.seed(42) # Fixed seed for reproducibility in evaluation? 
    # Actually, let's use different seeds for diversity
    for seed in range(10):
        v = np.random.normal(size=(N, 3), seed=seed)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        starts.append(v / norms)
        
    # 3. Perturbed Fibonacci
    for scale in [0.1, 0.05, 0.01]:
        noise = np.random.normal(size=(N, 3))
        p = fib_points + scale * noise
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        starts.append(p / norms)
        
    # Sort starts by initial energy (heuristic: better starts might lead to better ends, 
    # or we can just run them all. Let's run all within time budget)
    
    # Allocate time budget
    # We have ~145s. 
    # Let's give each local optimization ~10s if we have many starts.
    # Or run them sequentially until time runs out.
    
    for start_points in starts:
        if time.time() > DEADLINE - 5.0:
            break
            
        # Determine time limit for this optimization
        # If we have many starts, give less time to each?
        # Let's just run with a fixed limit, but check global deadline.
        remaining = DEADLINE - time.time()
        if remaining < 1.0:
            break
            
        # Use a portion of remaining time, but cap it to allow more restarts
        # If we have 14 starts, 145/14 ~ 10s each.
        # Let's use min(remaining * 0.9, 15.0)
        t_limit = min(remaining * 0.9, 15.0)
        
        opt_points, opt_energy = _optimize_local(start_points, time_limit=t_limit)
        
        if opt_energy < best_energy:
            best_points = opt_points
            best_energy = opt_energy
            
    return best_points
