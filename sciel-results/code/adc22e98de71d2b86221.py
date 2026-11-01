"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _log_energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dist, 1.0)
    # We only want upper triangle (i < j)
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Clamp to avoid log of very small numbers
    d = np.maximum(d, 1e-15)
    # Energy is sum of log(1/d) = -sum log(d)
    # Use math.fsum for precision as per task description
    log_vals = np.log(d)
    return -math.fsum(list(log_vals))

def _gradient_log_energy(points):
    """Compute gradient of the logarithmic energy with respect to points.
    
    Energy E = sum_{i<j} -log(|x_i - x_j|)
    dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(points)
    grad = np.zeros_like(points)
    # Vectorized computation for speed
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    # Avoid division by zero on diagonal
    dist_sq_diag = np.diag(dist_sq)
    np.fill_diagonal(dist_sq, 1.0) # Temporary fix to avoid 0/0 in division, we'll mask later
    
    # Gradient term: (x_i - x_j) / |x_i - x_j|^2
    # This is diff / dist_sq
    grad_terms = diff / dist_sq[:, :, np.newaxis]
    
    # Sum over j for each i
    # Note: The diagonal term was diff=0, so it contributes 0. 
    # But we replaced dist_sq diag with 1, so 0/1 = 0. Correct.
    grad = np.sum(grad_terms, axis=1)
    return grad

def _exponential_map(points, velocities, step_size):
    """Move points on the sphere using the exponential map.
    x_new = cos(step*|v|) * x + sin(step*|v|) * (v/|v|)
    """
    n = len(points)
    v_norms = np.linalg.norm(velocities, axis=1, keepdims=True)
    # Avoid division by zero
    v_norms_safe = np.where(v_norms < 1e-12, 1e-12, v_norms)
    
    cos_term = np.cos(step_size * v_norms_safe)
    sin_term = np.sin(step_size * v_norms_safe) / v_norms_safe
    
    # If v is 0, sin_term should be 0. 
    # With safe norm, if v_norm is tiny, sin(tiny)/tiny ~ 1. But v is 0, so 0*1=0.
    # Actually, if v is exactly 0, v_norm_safe is 1e-12. sin(1e-12*step)/1e-12 ~ step.
    # v * step = 0. So result is cos(0)*x + 0 = x. Correct.
    
    new_points = cos_term * points + sin_term * velocities
    return new_points

def construct():
    # Start with a good initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize to be safe
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    # Multistage Gradient Descent
    # Stage 1: Coarse optimization with larger steps to escape local minima
    # Stage 2: Fine optimization with smaller steps for precision
    
    # Adaptive step size or fixed schedule
    # Log energy is smooth but has singularities if points coincide.
    # We assume initial points are well separated.
    
    max_iters = 2000
    tolerance = 1e-10
    
    current_energy = _log_energy(points)
    
    # Learning rate schedule
    # Start with a moderate step, decrease over time
    initial_lr = 0.05
    final_lr = 1e-6
    decay_rate = 0.999
    
    lr = initial_lr
    momentum = 0.9
    velocity = np.zeros_like(points)
    
    for it in range(max_iters):
        grad = _gradient_log_energy(points)
        
        # Update velocity with momentum
        velocity = momentum * velocity - lr * grad
        
        # Move points
        new_points = _exponential_map(points, velocity, 1.0)
        
        # Ensure points are on sphere (exponential map should do this, but normalize to be safe)
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_energy = _log_energy(new_points)
        
        # If energy increases significantly, reject and reduce step size
        if new_energy > current_energy:
            lr *= 0.5
            # Revert
            continue
        else:
            points = new_points
            current_energy = new_energy
            # Decay learning rate
            lr *= decay_rate
            
            # Check convergence
            if lr < final_lr:
                break
                
            # Optional: check gradient norm for convergence
            # if np.linalg.norm(grad) < tolerance:
            #     break

    # Final refinement with very small steps to polish
    # Use a few iterations of small step gradient descent
    lr = 1e-7
    for it in range(100):
        grad = _gradient_log_energy(points)
        # Simple gradient step
        new_points = points - lr * grad
        # Project back to sphere
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_energy = _log_energy(new_points)
        if new_energy < current_energy:
            points = new_points
            current_energy = new_energy
        else:
            # If it increases, stop or reduce lr further
            lr *= 0.5
            if lr < 1e-12:
                break

    # Final normalization
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    # Check for close points
    min_dist = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_dist:
                min_dist = d
                
    if min_dist < 1e-9:
        # Perturb if necessary, though unlikely with this method
        rng = np.random.default_rng(42)
        points = points + rng.normal(0, 1e-6, points.shape)
        points /= np.linalg.norm(points, axis=1, keepdims=True)

    return points
# EVOLVE-BLOCK-END
