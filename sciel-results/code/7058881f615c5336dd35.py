"""Rewritten program: coupled variational optimization of log-energy on sphere."""
import numpy as np
from scipy.optimize import minimize
import math

N = 100


def _log_energy(points):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    # Compute all pairwise distances
    # Use efficient vectorized computation
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    # Mask diagonal
    np.fill_diagonal(dists, 1.0)
    # Compute log(1/d) = -log(d) for i<j
    # We need only upper triangle
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    # Clamp to avoid log(0)
    d = np.maximum(d, 1e-15)
    return -np.sum(np.log(d))


def _log_energy_grad(points):
    """Gradient of the logarithmic energy with respect to points."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 1.0)
    
    # Gradient: for each point i, sum over j != i of (x_i - x_j) / |x_i - x_j|^2 * ... 
    # Actually, d/dx_i [-log|x_i - x_j|] = -(x_i - x_j)/|x_i - x_j|^2
    # But we want gradient of sum_{i<j} -log|x_i-x_j|
    # For point k: sum over j != k of -(x_k - x_j)/|x_k - x_j|^2
    # Wait, let me be careful.
    # E = sum_{i<j} -log|x_i - x_j|
    # dE/dx_k = sum_{j != k} -d/dx_k log|x_k - x_j| (with appropriate sign)
    # If j < k: term is -log|x_k - x_j|, derivative w.r.t x_k is -(x_k - x_j)/|x_k - x_j|^2
    # If j > k: term is -log|x_j - x_k|, derivative w.r.t x_k is -(-1)*(x_j - x_k)/|x_j - x_k|^2 = (x_j - x_k)/|x_j - x_k|^2 = -(x_k - x_j)/|x_k - x_j|^2
    # So in both cases: dE/dx_k = -sum_{j != k} (x_k - x_j)/|x_k - x_j|^2
    
    grad = np.zeros_like(points)
    for k in range(n):
        for j in range(n):
            if j == k:
                continue
            d = dists[k, j]
            if d < 1e-15:
                continue
            grad[k] -= (points[k] - points[j]) / (d * d)
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
    
    # Project gradient onto tangent space of sphere
    # For each point, subtract radial component
    grad_tangent = np.zeros_like(grad_ambient)
    for i in range(N):
        p = points[i]
        g = grad_ambient[i]
        # Tangent component: g - (g·p)p
        radial = np.dot(g, p)
        grad_tangent[i] = g - radial * p
    
    return energy, grad_tangent.ravel()


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    
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
    
    # Use L-BFGS-B with projected gradient approach
    # We'll use a custom approach: optimize in ambient space, project after each step
    
    best_points = points0.copy()
    best_energy = _log_energy(best_points)
    
    # Multi-start optimization
    for restart in range(3):
        if restart > 0:
            # Different initializations
            if restart == 1:
                # Random initialization
                np.random.seed(42 + restart * 100)
                pts = np.random.randn(N, 3)
                pts = _project_to_sphere(pts)
            else:
                # Slightly perturbed Fibonacci
                np.random.seed(42 + restart * 100)
                pts = points0 + 0.1 * np.random.randn(N, 3)
                pts = _project_to_sphere(pts)
        else:
            pts = points0.copy()
        
        x_curr = pts.ravel()
        
        # Gradient descent with projected steps
        lr = 0.01
        for it in range(500):
            energy, grad_flat = _constrained_energy_and_grad(x_curr)
            grad = grad_flat.reshape(N, 3)
            
            # Check convergence
            if np.max(np.abs(grad)) < 1e-8:
                break
            
            # Take step in tangent direction
            x_new = x_curr - lr * grad_flat
            points_new = _project_to_sphere(x_new.reshape(N, 3))
            
            # Adaptive learning rate
            new_energy = _log_energy(points_new)
            if new_energy < energy:
                x_curr = points_new.ravel()
                # Increase learning rate slightly
                lr = min(lr * 1.05, 0.1)
            else:
                # Decrease learning rate
                lr = lr * 0.5
                # Try smaller step
                x_new2 = x_curr - lr * grad_flat
                points_new2 = _project_to_sphere(x_new2.reshape(N, 3))
                new_energy2 = _log_energy(points_new2)
                if new_energy2 < energy:
                    x_curr = points_new2.ravel()
            
            # Update best
            if new_energy < best_energy:
                best_energy = new_energy
                best_points = points_new.copy()
        
        # Update best from this restart
        final_energy = _log_energy(_project_to_sphere(x_curr.reshape(N, 3)))
        if final_energy < best_energy:
            best_energy = final_energy
            best_points = _project_to_sphere(x_curr.reshape(N, 3))
    
    # Final polish with scipy minimize using SLSQP on spherical constraint
    # We'll parameterize using quaternions or just use the projected approach
    
    # Do a final round of optimization
    x_final = best_points.ravel()
    
    for it in range(200):
        energy, grad_flat = _constrained_energy_and_grad(x_final)
        grad = grad_flat.reshape(N, 3)
        
        if np.max(np.abs(grad)) < 1e-9:
            break
        
        # Use a smaller learning rate for polishing
        lr = 0.001
        x_new = x_final - lr * grad_flat
        points_new = _project_to_sphere(x_new.reshape(N, 3))
        new_energy = _log_energy(points_new)
        
        if new_energy < energy:
            x_final = points_new.ravel()
            best_energy = new_energy
            best_points = points_new.copy()
        else:
            lr = 0.0001
            x_new = x_final - lr * grad_flat
            points_new = _project_to_sphere(x_new.reshape(N, 3))
            new_energy = _log_energy(points_new)
            if new_energy < energy:
                x_final = points_new.ravel()
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
