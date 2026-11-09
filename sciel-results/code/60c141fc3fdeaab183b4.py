"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.optimize import minimize

def _energy_and_gradient(pts):
    """Compute logarithmic energy and its gradient on the sphere.
    
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_k = sum_{j != k} (x_k - x_j) / |x_k - x_j|^2
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero
    np.fill_diagonal(dist_sq, 1.0)
    
    # Gradient: grad[k] = sum_{j != k} diff[k, j] / dist_sq[k, j]
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Energy
    i_idx, j_idx = np.triu_indices(n, k=1)
    dists = np.sqrt(dist_sq[i_idx, j_idx])
    dists = np.maximum(dists, 1e-300)
    E = -np.sum(np.log(dists))
    
    return E, grad

def _constraints(pts):
    """Define equality constraints: |x_i|^2 - 1 = 0 for each point."""
    cons = []
    for i in range(N):
        cons.append({
            'type': 'eq',
            'fun': lambda x, i=i: x[i*3:(i+1)*3] @ x[i*3:(i+1)*3] - 1.0,
            'jac': lambda x, i=i: np.concatenate([np.zeros(i*3), 2*x[i*3:(i+1)*3], np.zeros((N-1-i)*3)])
        })
    return cons

def _run_optimization(initial_pts):
    """Run SLSQP optimization from a given initial point."""
    x0 = initial_pts.flatten()
    cons = _constraints(initial_pts)
    
    def objective(x):
        pts = x.reshape(N, 3)
        E, _ = _energy_and_gradient(pts)
        return E
    
    def jac(x):
        pts = x.reshape(N, 3)
        _, grad = _energy_and_gradient(pts)
        return grad.flatten()
        
    try:
        res = minimize(objective, x0, method='SLSQP', jac=jac, constraints=cons, 
                       options={'maxiter': 500, 'ftol': 1e-12})
        if res.success:
            return res.fun, res.x.reshape(N, 3)
        else:
            # If it fails, return the current best guess from the optimization
            # SLSQP might return a point even if not 'success'
            # We should check if the constraints are satisfied
            pts = res.x.reshape(N, 3)
            # Normalize to ensure we return valid points
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            E, _ = _energy_and_gradient(pts)
            return E, pts
    except Exception as e:
        # Fallback to initial points if optimization crashes
        return _energy_and_gradient(initial_pts)[0], initial_pts

def construct():
    # Multiple starts to avoid local minima
    best_E = float('inf')
    best_pts = None
    
    # Start 1: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(pts_fib, axis=1, keepdims=True)
    pts_fib = pts_fib / norms
    
    # Start 2: Random
    rng = np.random.default_rng(42)
    pts_rand = rng.normal(size=(N, 3))
    norms = np.linalg.norm(pts_rand, axis=1, keepdims=True)
    pts_rand = pts_rand / norms
    
    # Start 3: Another random seed
    rng2 = np.random.default_rng(123)
    pts_rand2 = rng2.normal(size=(N, 3))
    norms = np.linalg.norm(pts_rand2, axis=1, keepdims=True)
    pts_rand2 = pts_rand2 / norms

    starts = [pts_fib, pts_rand, pts_rand2]
    
    for start_pts in starts:
        E, pts = _run_optimization(start_pts)
        if E < best_E:
            best_E = E
            best_pts = pts
    
    # Final normalization
    if best_pts is not None:
        norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
        best_pts = best_pts / norms
    
    return best_pts
# EVOLVE-BLOCK-END
