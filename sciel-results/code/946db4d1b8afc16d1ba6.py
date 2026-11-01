"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _compute_energy_and_grad(points):
    """
    Computes the logarithmic energy and its gradient for points on the sphere.
    Energy: sum_{i<j} log(1 / ||x_i - x_j||)
    Gradient: sum_{j != i} (x_j - x_i) / ||x_i - x_j||^2
    """
    # diff[i, j] = x_j - x_i
    diff = points[None, :, :] - points[:, None, :]
    d2 = np.sum(diff**2, axis=2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(d2, 1.0)
    
    # Gradient calculation
    # grad_i = sum_j (x_j - x_i) / d2_ij
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    grad = np.sum(diff * inv_d2[:, :, None], axis=1)
    
    # Energy calculation
    # We need sum_{i<j} log(1/sqrt(d2_ij)) = -0.5 * sum_{i != j} log(sqrt(d2_ij)) = -0.25 * sum_{i != j} log(d2_ij)
    # Or simply -0.5 * sum_{i<j} log(d_ij^2)
    # Let's compute log(d2) for off-diagonals
    log_d2 = np.log(d2)
    np.fill_diagonal(log_d2, 0.0)
    # sum_{i<j} log(1/d_ij) = sum_{i<j} -0.5 log(d2_ij)
    # sum_{i<j} A_ij = 0.5 * sum_{i != j} A_ij
    # So Energy = -0.5 * 0.5 * sum_{i != j} log(d2_ij) = -0.25 * sum_{i != j} log(d2_ij)
    energy = -0.25 * np.sum(log_d2)
    
    return energy, grad

def _project_to_sphere(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    return points / norms

def _riemannian_gradient_descent(points, max_iters=200, tol=1e-10):
    points = _project_to_sphere(points.copy())
    curr_energy, curr_grad = _compute_energy_and_grad(points)
    
    for it in range(max_iters):
        # Project gradient to tangent space
        # v_i = g_i - (g_i . x_i) x_i
        dot_gx = np.sum(curr_grad * points, axis=1, keepdims=True)
        tangent_grad = curr_grad - dot_gx * points
        
        # Check convergence
        grad_norm = np.linalg.norm(tangent_grad)
        if grad_norm < tol:
            break
        
        # Simple backtracking line search
        alpha = 0.01
        # Ensure step is not too large
        # Norm of tangent_grad can be large if points are close.
        # Scale step by inverse of grad norm if it's large?
        # Let's just try a few alphas.
        
        for _ in range(20):
            new_points = points - alpha * tangent_grad
            new_points = _project_to_sphere(new_points)
            new_energy, _ = _compute_energy_and_grad(new_points)
            
            if new_energy < curr_energy:
                points = new_points
                curr_energy = new_energy
                curr_grad, _ = _compute_energy_and_grad(points) # Recompute grad for next iter
                break
            else:
                alpha *= 0.5
        
        else:
            # If no step accepted, break or try smaller
            break
            
    return points, curr_energy

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Initializations
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = fib_points
    best_energy = float('inf')
    
    # Optimize Fibonacci start
    pts, e = _riemannian_gradient_descent(fib_points)
    if e < best_energy:
        best_energy = e
        best_points = pts
    
    # 2. Random Starts
    rng = np.random.default_rng(42)
    num_random_starts = 5
    
    for k in range(num_random_starts):
        if time.time() > deadline:
            break
        
        # Generate random points on sphere
        vecs = rng.normal(size=(N, 3))
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        
        pts, e = _riemannian_gradient_descent(vecs)
        if e < best_energy:
            best_energy = e
            best_points = pts

    # Final polish: try a few more iterations on the best point with tighter tolerance
    # We can just run the GD again on the best point
    if time.time() < deadline:
        pts, e = _riemannian_gradient_descent(best_points, max_iters=100, tol=1e-12)
        if e < best_energy:
            best_energy = e
            best_points = pts

    return _project_to_sphere(best_points)

# EVOLVE-BLOCK-END
