"""
Strategy: Repulsive Force Dynamics (RFD) on Cartesian Sphere with Projected Gradient Polish.
- Directly optimizes Cartesian coordinates on the sphere manifold.
- Uses the analytical gradient of the log-energy.
- Employs a momentum-based gradient descent with adaptive step size.
- Projects points back to the sphere after each step.
- Finishes with a high-precision projected gradient step for local refinement.
"""

import numpy as np
from math import fsum, log
import time

N = 150
DEADLINE_FRACTION = 0.90

def _energy_and_grad_cartesian(pts):
    """
    Compute log-energy and its gradient w.r.t. Cartesian coordinates.
    E = sum_{i<j} log(1/||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    Gradient:
    dE/dx_i = sum_{j!=i} -(x_i - x_j) / ||x_i - x_j||^2
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    
    mask = dists_sq > 1e-12
    inv_d_sq = np.zeros_like(dists_sq)
    inv_d_sq[mask] = 1.0 / dists_sq[mask]
    np.fill_diagonal(inv_d_sq, 0.0)
    
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(np.log(dists_sq[iu]))
    
    sum_inv_d_sq = np.sum(inv_d_sq, axis=1)
    grad = pts * sum_inv_d_sq[:, np.newaxis] - (inv_d_sq @ pts)
    
    return energy, grad

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    y = np.sqrt(1.0 - z**2) * np.cos(np.pi * (1.0 + np.sqrt(5.0)) * i)
    x = np.sqrt(1.0 - z**2) * np.sin(np.pi * (1.0 + np.sqrt(5.0)) * i)
    return np.column_stack([x, y, z])

def _random_points(n, rng):
    vecs = rng.standard_normal((n, 3))
    return _project_to_sphere(vecs)

def construct():
    start_time = time.time()
    total_time = 120.0
    deadline = start_time + total_time * DEADLINE_FRACTION
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_energy = 1e30
    
    n_restarts = 0
    
    while time.time() < deadline - 10:
        n_restarts += 1
        
        if n_restarts == 1:
            init_pts = _fibonacci_spiral(N)
        elif n_restarts % 2 == 0:
            init_pts = _random_points(N, rng)
        else:
            init_pts = _fibonacci_spiral(N)
            noise = rng.standard_normal((N, 3)) * 0.1
            init_pts = _project_to_sphere(init_pts + noise)
        
        max_steps = 500
        initial_step = 0.05
        final_step = 1e-5
        momentum = 0.9
        
        pts = init_pts.copy()
        v = np.zeros_like(pts)
        
        for k in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            energy, grad = _energy_and_grad_cartesian(pts)
            
            alpha = initial_step * (final_step / initial_step) ** (k / max_steps)
            
            v = momentum * v - alpha * grad
            pts = _project_to_sphere(pts + v)
        
        # Polish with projected gradient
        polish_steps = 100
        for k in range(polish_steps):
            if time.time() > deadline - 2:
                break
            energy, grad = _energy_and_grad_cartesian(pts)
            proj_grad = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
            alpha = 1e-4
            pts = _project_to_sphere(pts - alpha * proj_grad)
            
        final_energy, _ = _energy_and_grad_cartesian(pts)
        
        if final_energy < best_energy:
            best_energy = final_energy
            best_pts = pts.copy()
            
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
        
    best_pts = _project_to_sphere(best_pts)
    
    return best_pts
