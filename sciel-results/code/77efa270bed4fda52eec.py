import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def compute_energy(points):
    """
    Compute the logarithmic energy for N points on the sphere.
    Energy E = sum_{i<j} log(1 / |x_i - x_j|)
    """
    # points shape (N, 3)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero for i=j
    np.fill_diagonal(dists, 1.0)
    
    # Energy: sum_{i<j} log(1/dists[i,j])
    # log(1/d) = -log(d)
    # Sum over i!=j of -log(dists) / 2
    energy = -0.5 * np.sum(np.log(dists))
    
    return energy

def compute_energy_and_grads(points):
    """
    Compute the logarithmic energy and gradients for N points on the sphere.
    Energy E = sum_{i<j} log(1 / |x_i - x_j|)
    Gradient w.r.t x_i: sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    # points shape (N, 3)
    # diff shape (N, N, 3)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero for i=j
    np.fill_diagonal(dists, 1.0)
    
    # Energy: sum_{i<j} log(1/dists[i,j])
    energy = -0.5 * np.sum(np.log(dists))
    
    # Gradients: grad_i E = - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    dists_inv_sq = 1.0 / (dists * dists)
    np.fill_diagonal(dists_inv_sq, 0.0)
    
    # V_i = sum_j diff[i,j] * dists_inv_sq[i,j]
    # diff[i,j] = x_i - x_j
    # V = np.einsum('ijk,ij->ik', diff, dists_inv_sq)
    # grad E = -V
    grads = -np.einsum('ijk,ij->ik', diff, dists_inv_sq)
    
    return energy, grads

def objective_and_grad(x_flat):
    """
    Objective function and gradient for L-BFGS-B.
    x_flat: flat array of shape (N*3,)
    """
    points = x_flat.reshape(N, 3)
    energy, grads = compute_energy_and_grads(points)
    return energy, grads.flatten()

def project_to_sphere(points):
    """Project points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def construct():
    start_time = time.time()
    time_limit = 145.0  # 120s limit, leave margin for safety
    
    # Best solution tracking
    best_points = None
    best_energy = np.inf
    
    def try_optimize(initial_points, max_time):
        nonlocal best_points, best_energy
        
        # Flatten for optimization
        x0 = initial_points.flatten()
        
        # Use L-BFGS-B with custom objective
        # We'll do gradient descent on the sphere manually with L-BFGS-like behavior
        # Actually, let's use a simpler approach: gradient descent with adaptive step size
        
        points = initial_points.copy()
        current_energy = compute_energy(points)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_points = points.copy()
        
        # Adaptive gradient descent with momentum
        lr = 1e-3
        momentum = 0.9
        v = np.zeros_like(points)
        
        max_iters = 5000
        no_improve_count = 0
        best_local_energy = current_energy
        best_local_points = points.copy()
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            energy, grads = compute_energy_and_grads(points)
            
            if not np.isfinite(energy):
                break
            
            # Update momentum
            v = momentum * v - lr * grads
            
            # Update points
            new_points = points + v
            
            # Project to sphere
            norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            new_points = new_points / norms
            
            # Compute new energy
            new_energy = compute_energy(new_points)
            
            if not np.isfinite(new_energy):
                # Revert
                lr *= 0.5
                v = np.zeros_like(points)
                continue
            
            if new_energy < best_local_energy:
                best_local_energy = new_energy
                best_local_points = new_points.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if new_energy < current_energy:
                points = new_points
                current_energy = new_energy
            else:
                # Reduce learning rate and reset momentum
                lr *= 0.7
                v = 0.5 * v
            
            # Update global best
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = points.copy()
            
            # Check convergence
            if it > 20 and no_improve_count > 100:
                break
            
            # Adaptive learning rate increase if improving consistently
            if no_improve_count == 0 and lr < 1e-2:
                lr = min(lr * 1.1, 1e-2)
        
        return best_local_points, best_local_energy
    
    # Initial Fibonacci spiral
    def fibonacci_spiral():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Try multiple initializations
    initializations = []
    
    # 1. Fibonacci spiral
    initializations.append(fibonacci_spiral())
    
    # 2. Random points (several attempts)
    rng = np.random.default_rng(42)
    for _ in range(5):
        pts = rng.standard_normal((N, 3))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        initializations.append(pts / norms)
    
    # 3. Perturbed Fibonacci
    fib = fibonacci_spiral()
    for _ in range(3):
        perturbed = fib + rng.standard_normal((N, 3)) * 0.1
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        initializations.append(perturbed / norms)
    
    # Optimize each initialization
    for init_idx, init_points in enumerate(initializations):
        if time.time() - start_time > time_limit:
            break
        
        # Allocate time based on remaining budget
        remaining = time_limit - (time.time() - start_time)
        max_time_for_this = remaining / (len(initializations) - init_idx)
        
        try:
            local_points, local_energy = try_optimize(init_points, max_time_for_this)
        except Exception:
            continue
    
    # Final check: ensure best_points is valid
    if best_points is None:
        # Fallback to Fibonacci spiral
        best_points = fibonacci_spiral()
    
    # Ensure all points are on unit sphere and non-zero
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_points = best_points / norms
    
    return best_points
