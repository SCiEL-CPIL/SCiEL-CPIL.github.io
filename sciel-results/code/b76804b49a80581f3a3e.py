import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 145.0  # Safety margin from 150s limit

def _energy_grad_riemannian(pts):
    """
    Compute Riemannian energy and gradient on the sphere.
    E = sum_{i<j} log(1/||x_i - x_j||)
    Ambient gradient: dE/dx_i = -sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    Riemannian gradient: Proj_{T_x S^2} (dE/dx_i) = dE/dx_i - <dE/dx_i, x_i> x_i
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists2, 1.0) # Avoid division by zero
    
    # Energy: -0.5 * sum log(d^2) over upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(np.log(dists2[iu]))
    
    # Ambient Gradient
    inv_dists2 = 1.0 / dists2
    np.fill_diagonal(inv_dists2, 0.0)
    
    # grad_amb[i] = -sum_j (x_i - x_j) / d_ij^2
    # Note: diff[i,j] = x_i - x_j
    # sum_j diff[i,j] * inv_dists2[i,j]
    ambient_grad = -np.sum(diff * inv_dists2[:, :, np.newaxis], axis=1)
    
    # Project to Tangent Space
    # <grad, x> is a scalar for each point
    proj_coeffs = np.sum(ambient_grad * pts, axis=1, keepdims=True)
    riemannian_grad = ambient_grad - proj_coeffs * pts
    
    return energy, riemannian_grad

def _project(pts):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _riemannian_descent(pts, time_budget, initial_lr=0.05, min_lr=1e-10):
    """
    Riemannian Gradient Descent with backtracking line search.
    """
    start_time = time.time()
    pts = _project(pts)
    current_energy, current_grad = _energy_grad_riemannian(pts)
    
    lr = initial_lr
    patience = 0
    max_patience = 100
    
    while time.time() - start_time < time_budget:
        grad_norm = np.linalg.norm(current_grad)
        if grad_norm < 1e-12:
            break
            
        # Backtracking line search
        alpha = lr
        found = False
        for _ in range(30):
            # Retraction: move in direction -grad, then project
            new_pts = pts - alpha * current_grad
            new_pts = _project(new_pts)
            
            new_energy, new_grad = _energy_grad_riemannian(new_pts)
            
            # Armijo condition (simplified)
            if new_energy < current_energy - 1e-4 * alpha * grad_norm:
                pts = new_pts
                current_energy = new_energy
                current_grad = new_grad
                found = True
                break
            alpha *= 0.5
        
        if found:
            patience = 0
            # Increase LR slightly if step was successful and large
            if alpha > 0.1:
                lr = min(alpha * 1.5, 0.5)
        else:
            patience += 1
            lr = alpha
            if patience > max_patience or lr < min_lr:
                break
                
    return pts

def _lbfgs_polish(pts, time_budget):
    """
    Final polish using L-BFGS-B on the ambient space.
    This is often very effective for high-precision local minima.
    """
    start_time = time.time()
    pts = _project(pts)
    x0 = pts.flatten()
    
    def func_and_grad(x):
        p = _project(x.reshape(-1, 3))
        e, g = _energy_grad_riemannian(p)
        # For L-BFGS-B, we can use the ambient gradient or projected.
        # Using projected gradient is safer for the manifold constraint, 
        # but L-BFGS-B doesn't know about the manifold.
        # We will use the ambient gradient derived from the Riemannian one?
        # Actually, standard L-BFGS-B on ambient space with projection inside the function
        # is a common heuristic. 
        # Let's use the ambient gradient for the Jacobian to help the Hessian approximation.
        # Recompute ambient grad for L-BFGS-B
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dists2, 1.0)
        inv_dists2 = 1.0 / dists2
        np.fill_diagonal(inv_dists2, 0.0)
        ambient_grad = -np.sum(diff * inv_dists2[:, :, np.newaxis], axis=1)
        return e, ambient_grad.flatten()
        
    options = {
        'maxiter': 5000,
        'gtol': 1e-14,
        'ftol': 1e-18,
        'disp': False,
        'maxfun': 20000,
        'maxls': 50
    }
    
    try:
        result = minimize(func_and_grad, x0, jac=True, method='L-BFGS-B', options=options)
        if result.fun < 1e17:
            return _project(result.x.reshape(-1, 3))
    except Exception:
        pass
        
    return pts

def _fibonacci_spiral(n=N):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n=N, seed=None):
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    return _project(pts)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_pts = None
    best_energy = float('inf')
    
    # Generate diverse starting points
    starts = [
        _fibonacci_spiral(N),
        _project(_fibonacci_spiral(N) + np.random.default_rng(1).normal(0, 0.1, (N, 3))),
        _project(_fibonacci_spiral(N) + np.random.default_rng(2).normal(0, 0.2, (N, 3))),
        _random_sphere(N, seed=10),
        _random_sphere(N, seed=20),
        _random_sphere(N, seed=30),
        _random_sphere(N, seed=40),
        _random_sphere(N, seed=50),
    ]
    
    # Filter similar starts to save time
    unique_starts = [starts[0]]
    for s in starts[1:]:
        if all(np.linalg.norm(s - u) > 0.5 for u in unique_starts):
            unique_starts.append(s)
    starts = unique_starts
    
    # Allocate time: 70% for Riemannian descent, 30% for L-BFGS polish
    total_time = deadline - start_time
    descent_deadline = start_time + total_time * 0.7
    
    for i, start in enumerate(starts):
        if time.time() > descent_deadline:
            break
        
        # Riemannian Descent
        # Give more time to the first few starts as they are likely better
        time_budget = min(10.0, (descent_deadline - time.time()) / (len(starts) - i))
        if time_budget < 1.0:
            break
            
        optimized = _riemannian_descent(start, time_budget)
        e, _ = _energy_grad_riemannian(optimized)
        
        if e < best_energy:
            best_energy = e
            best_pts = optimized
            
    # Final Polish with L-BFGS-B
    if best_pts is not None:
        polish_time = deadline - time.time() - 2.0 # 2s safety margin
        if polish_time > 1.0:
            polished = _lbfgs_polish(best_pts, polish_time)
            e, _ = _energy_grad_riemannian(polished)
            if e < best_energy:
                best_energy = e
                best_pts = polished
        
        # Final Projection and Validation
        best_pts = _project(best_pts)
        
        # Check for invalid points (too close)
        diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        
        if min_d < 1e-9:
            # Fallback to Fibonacci if invalid
            return _fibonacci_spiral(N)
            
        # Ensure non-finite values are handled
        if not np.all(np.isfinite(best_pts)):
            return _fibonacci_spiral(N)
            
    else:
        best_pts = _fibonacci_spiral(N)
        
    return best_pts
