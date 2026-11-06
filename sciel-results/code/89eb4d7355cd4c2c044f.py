"""Rewritten program: spectral deformation + Riemannian polish for log-energy minimization."""
import numpy as np
import math
from scipy.optimize import minimize

N = 150

def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere.
    Energy: sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)
    Gradient wrt xi: sum_{j!=i} (xi - xj) / |xi-xj|^2
    """
    n = len(pts)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n,n,3)
    dist = np.linalg.norm(diff, axis=2)  # (n,n)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy: -sum_{i<j} log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each i, grad_i = sum_{j!=i} (xi - xj) / |xi-xj|^2
    # diff[i,j] = xi - xj
    # grad[i] = sum_j (xi - xj) / dist[i,j]^2
    inv_dist2 = 1.0 / (dist ** 2)
    np.fill_diagonal(inv_dist2, 0.0)
    grad = np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure points are on sphere
    pts = _project_to_sphere(pts)
    
    # Riemannian gradient descent with backtracking line search
    # We work in ambient space but project back to sphere after each step.
    
    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(best_pts)
    
    lr = 0.01
    max_iter = 5000
    patience = 500
    no_improve = 0
    
    for it in range(max_iter):
        energy, grad = _energy_and_grad(pts)
        
        # Project gradient to tangent space: grad_t = grad - (grad . x) * x
        grad_proj = grad - np.sum(pts * grad, axis=1, keepdims=True) * pts
        
        # Backtracking line search
        step_lr = lr
        for _ in range(20):
            new_pts = pts + step_lr * grad_proj
            new_pts = _project_to_sphere(new_pts)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy - 1e-10:
                break
            step_lr *= 0.5
        
        else:
            # No improvement found, try smaller step
            step_lr = lr * 1e-3
            new_pts = pts + step_lr * grad_proj
            new_pts = _project_to_sphere(new_pts)
            new_energy, _ = _energy_and_grad(new_pts)
        
        if new_energy < energy - 1e-10:
            pts = new_pts
            if new_energy < best_energy - 1e-12:
                best_energy = new_energy
                best_pts = pts.copy()
                no_improve = 0
            else:
                no_improve += 1
            # Slowly decrease learning rate
            if it % 200 == 0 and it > 0:
                lr *= 0.9
        else:
            no_improve += 1
            lr *= 0.95
        
        if no_improve > patience:
            break
    
    pts = best_pts
    
    # Final polish with smaller steps
    lr = 0.001
    for it in range(2000):
        energy, grad = _energy_and_grad(pts)
        grad_proj = grad - np.sum(pts * grad, axis=1, keepdims=True) * pts
        
        step_lr = lr
        for _ in range(20):
            new_pts = pts + step_lr * grad_proj
            new_pts = _project_to_sphere(new_pts)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy - 1e-12:
                break
            step_lr *= 0.5
        
        else:
            break
        
        if new_energy < energy - 1e-12:
            pts = new_pts
            if it % 100 == 0:
                lr *= 0.9
        else:
            break
    
    # Final projection and validation
    pts = _project_to_sphere(pts)
    
    # Check for very close points
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # Perturb points that are too close
        for k in range(100):
            iu = np.argwhere(dist < 1e-9)
            if len(iu) == 0:
                break
            idx = iu[0]
            # Random small perturbation
            noise = np.random.randn(3) * 1e-6
            pts[idx[0]] += noise
            pts[idx[0]] /= np.linalg.norm(pts[idx[0]])
            pts[idx[1]] += noise
            pts[idx[1]] /= np.linalg.norm(pts[idx[1]])
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, 1.0)
    
    return pts
