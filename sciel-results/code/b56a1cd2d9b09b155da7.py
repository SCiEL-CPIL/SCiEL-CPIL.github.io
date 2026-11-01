"""Optimized logarithmic energy minimizer for 100 points on the unit sphere."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from math import fsum, log


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on the sphere."""
    n = pts.shape[0]
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    energy = -np.sum(np.log(d))
    
    # Gradient: dE/dpts_i = sum_{j != i} (pts_i - pts_j) / |pts_i - pts_j|^2
    # For each point i, grad_i = sum_{j != i} (pts_i - pts_j) / dist_ij^2
    # Note: d/dx_i [ -log(|x_i - x_j|) ] = -(x_i - x_j)/|x_i - x_j|^2
    # Wait, let me re-derive. E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    # dE/dx_k = -sum_{j != k} d/dx_k log(|x_k - x_j|) = -sum_{j != k} (x_k - x_j)/|x_k - x_j|^2
    # So grad_k = -sum_{j != k} (x_k - x_j)/dist_kj^2
    # Which is sum_{j != k} (x_j - x_k)/dist_kj^2
    
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d_ij = dist[i, j]
            if d_ij < 1e-15:
                continue
            grad[i] += (pts[j] - pts[i]) / (d_ij * d_ij)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    np.random.seed(42)
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Use L-BFGS-B with sphere constraint via projection
    # We'll optimize in R^3 and project back to sphere after each step
    # Actually, let's use a parameterization that stays on the sphere
    # Or use scipy minimize with bounds and then project
    
    # Better approach: use multiple restarts with different initializations
    best_pts = pts.copy()
    best_energy = _energy_and_grad(best_pts)[0]
    
    # Try several different initializations and optimize each
    n_restarts = 10
    for restart in range(n_restarts):
        if restart == 0:
            current_pts = pts.copy()
        else:
            # Random initialization
            current_pts = np.random.randn(N, 3)
            current_pts = _project_to_sphere(current_pts)
        
        # Use L-BFGS-B optimization
        # We need to handle the sphere constraint. One approach:
        # Optimize in ambient space with a penalty for deviating from sphere
        # Or use a reparameterization
        
        # Let's use a simple approach: gradient descent with projection
        # Compute gradient and do projected gradient descent
        
        lr = 0.01
        for step in range(2000):
            energy, grad = _energy_and_grad(current_pts)
            
            # Project gradient to tangent space of sphere
            # Tangent projection: grad_tangent = grad - (grad . x) * x
            grad_proj = grad - np.sum(grad * current_pts, axis=1, keepdims=True) * current_pts
            
            # Update
            current_pts = current_pts - lr * grad_proj
            
            # Project back to sphere
            current_pts = _project_to_sphere(current_pts)
            
            # Adaptive learning rate
            new_energy = _energy_and_grad(current_pts)[0]
            if new_energy > energy:
                lr *= 0.5
                current_pts = current_pts + lr * grad_proj
                current_pts = _project_to_sphere(current_pts)
                new_energy = _energy_and_grad(current_pts)[0]
            else:
                # Occasionally increase lr
                if step % 100 == 0 and lr < 0.1:
                    lr *= 1.1
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = current_pts.copy()
        
        # Local refinement with smaller steps
        lr = 0.001
        for step in range(1000):
            energy, grad = _energy_and_grad(best_pts)
            grad_proj = grad - np.sum(grad * best_pts, axis=1, keepdims=True) * best_pts
            best_pts = best_pts - lr * grad_proj
            best_pts = _project_to_sphere(best_pts)
            new_energy = _energy_and_grad(best_pts)[0]
            if new_energy > energy:
                lr *= 0.5
                best_pts = best_pts + lr * grad_proj
                best_pts = _project_to_sphere(best_pts)
    
    # Final cleanup: ensure points are on sphere and well-separated
    best_pts = _project_to_sphere(best_pts)
    
    # Check for any very close points and perturb if needed
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Perturb point j
                best_pts[j] = best_pts[j] + 1e-6 * np.random.randn(3)
                best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
