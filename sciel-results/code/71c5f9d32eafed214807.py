"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 115.0  # safety margin before 120s kill

# EVOLVE-BLOCK-START

def _log_energy_and_grad(pts):
    """Compute logarithmic energy and its gradient.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    # pts shape (N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    
    # Avoid division by zero (diagonal)
    np.fill_diagonal(dist_sq, 1.0)
    
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of -log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu[0], iu[1]]))
    
    # Gradient: for each i, grad_i = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j
    # grad_i = -sum_j diff[i,j] / dist_sq[i,j]
    inv_dist_sq = 1.0 / dist_sq  # (N, N)
    # Set diagonal to 0
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = -sum_j diff[i,j] * inv_dist_sq[i,j]
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)  # (N, 3)
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.sqrt(np.sum(pts * pts, axis=1, keepdims=True))
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_spiral():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_start(seed):
    """Generate random points on the sphere."""
    rng = np.random.RandomState(seed)
    theta = rng.uniform(0, 2 * np.pi, N)
    phi = np.arccos(rng.uniform(-1, 1, N))
    pts = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    return pts

def _riemannian_gradient_descent(pts0, deadline, max_iter=1000):
    """
    Custom Riemannian gradient descent with Armijo line search.
    This is used for final polishing to achieve high precision.
    """
    x = pts0.copy()
    best_energy = _log_energy_and_grad(x)[0]
    best_pts = x.copy()
    
    lr = 0.01
    beta = 0.9
    v = np.zeros_like(x)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        energy, grad = _log_energy_and_grad(x)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = x.copy()
            
        # Project gradient to tangent space
        grad_tan = grad - np.sum(grad * x, axis=1, keepdims=True) * x
        
        # Momentum
        v = beta * v - lr * grad_tan
        
        # Exponential map approximation (for small steps, x + v is close)
        # But to be safe, we project
        x_new = _project_to_sphere(x + v)
        
        # Armijo line search
        new_energy = _log_energy_and_grad(x_new)[0]
        slope = np.sum(grad_tan * grad_tan)
        
        if new_energy <= energy - 1e-4 * lr * slope:
            x = x_new
        else:
            # Halve step
            v = 0.5 * v
            x = _project_to_sphere(x + v)
            new_energy = _log_energy_and_grad(x)[0]
            if new_energy > energy:
                # Revert if it got worse
                v = 0.5 * v
                x = _project_to_sphere(x + v)
        
        # Decay learning rate
        lr = max(0.001, lr * 0.999)
        
    return best_pts, best_energy

def _optimize_from_start(pts0, deadline):
    """Run L-BFGS-B optimization from a starting point."""
    best_pts = pts0.copy()
    best_energy = _log_energy_and_grad(pts0)[0]
    
    # L-BFGS-B
    try:
        res = minimize(
            fun=lambda x: _log_energy_and_grad(x)[0],
            x0=pts0.ravel(),
            jac=lambda x: _log_energy_and_grad(x)[1].ravel(),
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
        )
        if res.x is not None and np.all(np.isfinite(res.x)):
            cand = _project_to_sphere(res.x.reshape(N, 3))
            e = _log_energy_and_grad(cand)[0]
            if e < best_energy and np.all(np.isfinite(cand)):
                best_pts = cand
                best_energy = e
    except Exception:
        pass
    
    # Polish with Riemannian GD if time remains
    if time.time() < deadline - 2:
        pts, e = _riemannian_gradient_descent(best_pts, deadline, max_iter=200)
        if e < best_energy:
            best_pts = pts
            best_energy = e
            
    return best_pts, best_energy

def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    global DEADLINE
    deadline = DEADLINE
    
    # Start with Fibonacci spiral
    best_pts = _fibonacci_spiral()
    best_energy = _log_energy_and_grad(best_pts)[0]
    
    # Optimize from Fibonacci start
    pts, energy = _optimize_from_start(best_pts, deadline - 5)
    if energy < best_energy:
        best_pts = pts
        best_energy = energy
        
    # Try a few random starts if time permits
    t_start = time.time()
    for seed in range(5):
        if time.time() > deadline - 10:
            break
        pts0 = _random_start(seed)
        pts, energy = _optimize_from_start(pts0, deadline - 5)
        if energy < best_energy:
            best_pts = pts
            best_energy = energy
            
    # Final polish on the best point found
    if time.time() < deadline - 1:
        pts, energy = _riemannian_gradient_descent(best_pts, deadline, max_iter=500)
        if energy < best_energy:
            best_pts = pts
            best_energy = energy

    # Ensure validity
    best_pts = _project_to_sphere(best_pts)
    
    # Check for close points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                best_pts[j] += np.random.randn(3) * 1e-10
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
                
    return best_pts

# EVOLVE-BLOCK-END
