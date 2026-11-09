"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _log_energy(pts):
    """Compute sum_{i<j} log(1/|x_i - x_j|) using math.fsum for accuracy."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return -1e18  # penalty for overlapping points
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _log_energy_fast(pts):
    """Vectorized log energy for speed during optimization."""
    n = len(pts)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Mask diagonal and upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    valid_dists = dists[mask]
    if np.any(valid_dists < 1e-12):
        return -1e18
    return np.sum(np.log(1.0 / valid_dists))

def _energy_and_grad(pts):
    """Compute log energy and its gradient with respect to points on sphere."""
    n = len(pts)
    energy = 0.0
    grad = np.zeros_like(pts)
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-12:
                return -1e18, grad
            # d/dx_i log(1/|x_i - x_j|) = -1/d * (x_i - x_j)/|x_i - x_j| = -diff/d^2
            # Wait: d/dx log(1/|x-y|) = -1/|x-y| * (x-y)/|x-y|^2 * ... 
            # Let f = -log|diff|. df/dx = -1/|diff| * diff/|diff| = -diff/|diff|^2
            # So gradient contribution to x_i: -diff/d^2, to x_j: +diff/d^2
            energy += math.log(1.0 / d)
            grad[i] -= diff / (d * d)
            grad[j] += diff / (d * d)
    
    return energy, grad

def _project_to_tangent(pts, vec):
    """Project vec onto tangent space at each point."""
    return vec - np.sum(vec * pts, axis=1, keepdims=True) * pts

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    """Generate random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _gradient_descent(pts, max_iter=5000, tol=1e-12, lr_schedule='adaptive'):
    """Gradient descent on the sphere with adaptive step size."""
    pts = pts.copy()
    n = len(pts)
    
    # Initial learning rate
    lr = 0.1
    best_energy = _log_energy_fast(pts)
    best_pts = pts.copy()
    stale = 0
    
    for it in range(max_iter):
        # Compute energy and gradient
        energy, grad = _energy_and_grad(pts)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            stale = 0
        else:
            stale += 1
        
        # Project gradient to tangent space
        grad_tan = _project_to_tangent(pts, grad)
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_tan)
        if grad_norm < tol:
            break
        
        # Adaptive step size
        if lr_schedule == 'adaptive':
            # Scale step by inverse gradient norm for stability
            step = lr / max(1.0, grad_norm)
        else:
            step = lr
        
        # Update
        pts_new = pts + step * grad_tan
        
        # Renormalize to sphere
        norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
        pts_new = pts_new / norms
        
        # Check if energy improved
        new_energy = _log_energy_fast(pts_new)
        
        if new_energy < energy:
            pts = pts_new
            # Increase learning rate slightly if improving
            if stale == 0:
                lr = min(lr * 1.05, 1.0)
        else:
            # Decrease learning rate
            lr *= 0.5
            if lr < 1e-8:
                break
        
        if stale > 50:
            break
    
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin
    
    # Start with Fibonacci spiral
    pts0 = _fibonacci_sphere(N)
    best_pts, best_energy = _gradient_descent(pts0, max_iter=10000, tol=1e-13)
    
    # Try multiple random restarts
    rng = np.random.RandomState(42)
    n_restarts = 20
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        pts_r = _random_sphere(N, seed=rng.randint(1000000))
        
        # Gradient descent
        pts_opt, energy_opt = _gradient_descent(pts_r, max_iter=5000, tol=1e-13)
        
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Final high-precision polish
    if time.time() < deadline - 10:
        best_pts, best_energy = _gradient_descent(best_pts, max_iter=20000, tol=1e-15)
    
    # Verify and return
    pts = best_pts.copy()
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    # Final check
    final_energy = _log_energy(pts)
    
    return pts

# EVOLVE-BLOCK-END
