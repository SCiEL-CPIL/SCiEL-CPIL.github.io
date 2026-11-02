"""Rewritten program: coupled variational optimization of log-energy on sphere."""
import numpy as np
from scipy.optimize import minimize
import math
import time

N = 100


def _log_energy(points):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    # Compute all pairwise distances using vectorized computation
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    # Mask diagonal
    np.fill_diagonal(dists_sq, 1.0)
    # Compute log(1/d) = -log(d) for i<j
    iu = np.triu_indices(n, k=1)
    d_sq = dists_sq[iu]
    # Clamp to avoid log(0)
    d_sq = np.maximum(d_sq, 1e-30)
    d = np.sqrt(d_sq)
    return -math.fsum(np.log(d))


def _log_energy_grad(points):
    """Gradient of the logarithmic energy with respect to points (vectorized)."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    
    # dists_sq[i,j] = |x_i - x_j|^2
    # For each point k, grad[k] = -sum_{j != k} (x_k - x_j) / |x_k - x_j|^2
    # = -sum_{j != k} diff[k,j] / dists_sq[k,j]
    
    # diff / dists_sq[:, :, np.newaxis] gives diff[k,j] / dists_sq[k,j] for each component
    inv_dists_sq = 1.0 / np.maximum(dists_sq, 1e-30)
    # Set diagonal to 0
    np.fill_diagonal(inv_dists_sq, 0.0)
    
    # grad[k] = -sum_j diff[k,j] * inv_dists_sq[k,j]
    # diff[k,j] = points[k] - points[j]
    # So grad[k] = -sum_j (points[k] - points[j]) * inv_dists_sq[k,j]
    #           = -points[k] * sum_j inv_dists_sq[k,j] + sum_j points[j] * inv_dists_sq[k,j]
    
    # Vectorized: 
    # sum_j inv_dists_sq[k,j] -> row sums
    row_sums = np.sum(inv_dists_sq, axis=1)
    # sum_j points[j] * inv_dists_sq[k,j] -> matrix-vector product
    weighted_points = inv_dists_sq @ points
    
    grad = -points * row_sums[:, np.newaxis] + weighted_points
    return grad


def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _constrained_energy_and_grad(x):
    """
    x: flattened array of 3*N coordinates.
    Returns energy and gradient for points on sphere (using Lagrange multiplier approach 
    or just project after gradient).
    """
    points = x.reshape(N, 3)
    # Project to sphere for evaluation
    points = _project_to_sphere(points)
    energy = _log_energy(points)
    
    # Compute gradient in ambient space
    grad_ambient = _log_energy_grad(points)
    
    # Project gradient onto tangent space of sphere (vectorized)
    # For each point i: grad_tangent[i] = grad_ambient[i] - (grad_ambient[i] · points[i]) * points[i]
    radial_components = np.sum(grad_ambient * points, axis=1, keepdims=True)
    grad_tangent = grad_ambient - radial_components * points
    
    return energy, grad_tangent.ravel()


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    points0 = points0 + 0.01 * np.random.randn(N, 3)
    points0 = _project_to_sphere(points0)
    
    x0 = points0.ravel()
    
    best_points = points0.copy()
    best_energy = _log_energy(best_points)
    
    # Multiple restarts with different initializations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
        
        if restart == 0:
            pts = points0.copy()
        elif restart == 1:
            # Random initialization
            np.random.seed(42 + restart * 100)
            pts = np.random.randn(N, 3)
            pts = _project_to_sphere(pts)
        elif restart == 2:
            # Slightly perturbed Fibonacci
            np.random.seed(42 + restart * 100)
            pts = points0 + 0.1 * np.random.randn(N, 3)
            pts = _project_to_sphere(pts)
        elif restart == 3:
            # Different Fibonacci with different seed
            np.random.seed(42 + restart * 100)
            i2 = np.arange(N) + 0.5
            z2 = 1.0 - 2.0 * i2 / N
            phi2 = np.pi * (1.0 + np.sqrt(5.0)) * (i2 + 0.5)
            r2 = np.sqrt(1.0 - z2 * z2)
            pts = np.column_stack([r2 * np.cos(phi2), r2 * np.sin(phi2), z2])
            pts = pts + 0.02 * np.random.randn(N, 3)
            pts = _project_to_sphere(pts)
        else:
            # Perturbed best so far
            np.random.seed(42 + restart * 100)
            pts = best_points + 0.05 * np.random.randn(N, 3)
            pts = _project_to_sphere(pts)
        
        x_curr = pts.ravel()
        
        # Use L-BFGS-B with projected gradient
        # We'll do gradient descent with adaptive step size
        lr = 0.01
        prev_energy = None
        stagnation_count = 0
        
        for it in range(300):
            if time.time() - start_time > time_limit:
                break
            
            energy, grad_flat = _constrained_energy_and_grad(x_curr)
            grad = grad_flat.reshape(N, 3)
            
            # Check convergence
            if np.max(np.abs(grad)) < 1e-9:
                break
            
            # Adaptive learning rate with backtracking
            step_size = lr
            accepted = False
            for _ in range(10):
                x_new = x_curr - step_size * grad_flat
                points_new = _project_to_sphere(x_new.reshape(N, 3))
                new_energy = _log_energy(points_new)
                
                if new_energy < energy - 1e-10:
                    x_curr = points_new.ravel()
                    accepted = True
                    break
                else:
                    step_size *= 0.5
            
            if not accepted:
                stagnation_count += 1
                if stagnation_count > 20:
                    break
                lr = lr * 0.5
            else:
                stagnation_count = 0
                # Increase learning rate slightly if we've been successful
                if it > 50:
                    lr = min(lr * 1.02, 0.05)
            
            # Update best
            if new_energy < best_energy:
                best_energy = new_energy
                best_points = points_new.copy()
        
        # Update best from this restart
        final_points = _project_to_sphere(x_curr.reshape(N, 3))
        final_energy = _log_energy(final_points)
        if final_energy < best_energy:
            best_energy = final_energy
            best_points = final_points
    
    # Final polish with smaller steps
    x_final = best_points.ravel()
    lr = 0.001
    for it in range(100):
        if time.time() - start_time > time_limit:
            break
        
        energy, grad_flat = _constrained_energy_and_grad(x_final)
        grad = grad_flat.reshape(N, 3)
        
        if np.max(np.abs(grad)) < 1e-10:
            break
        
        step_size = lr
        accepted = False
        for _ in range(15):
            x_new = x_final - step_size * grad_flat
            points_new = _project_to_sphere(x_new.reshape(N, 3))
            new_energy = _log_energy(points_new)
            
            if new_energy < energy - 1e-12:
                x_final = points_new.ravel()
                accepted = True
                break
            else:
                step_size *= 0.5
        
        if not accepted:
            break
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_points = points_new.copy()
    
    # Ensure valid output
    result = _project_to_sphere(best_points)
    
    # Validate: no points too close, all finite
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(result[i] - result[j])
            if d < 1e-9:
                # Perturb slightly
                np.random.seed(12345 + i * 100 + j)
                result[j] += 1e-8 * np.random.randn(3)
                result[j] = result[j] / np.linalg.norm(result[j])
    
    if not np.all(np.isfinite(result)):
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        result = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return result
