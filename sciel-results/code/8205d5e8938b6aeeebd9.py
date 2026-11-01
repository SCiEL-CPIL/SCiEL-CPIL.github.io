"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _potential_and_grad(x):
    """
    Calculate potential energy and gradient for sphere packing.
    x is flattened array of shape (3*N,).
    """
    c = x.reshape(N, 3)
    V = 0.0
    grad = np.zeros_like(x)
    
    # Sphere-sphere repulsion
    # diff: (N, N, 3)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=2)
    # Avoid division by zero on diagonal
    np.fill_diagonal(d2, 1.0)
    d2_inv = 1.0 / d2
    np.fill_diagonal(d2_inv, 0.0)
    
    V += np.sum(d2_inv)
    
    # Gradient contribution from sphere-sphere
    # grad_i = sum_j -2/d2^2 * (c_i - c_j)
    factor = -2.0 * d2_inv**2
    np.fill_diagonal(factor, 0.0)
    
    grad_3d = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    grad[:] = grad_3d.flatten()
    
    # Sphere-wall repulsion
    # V += sum_k 1/(0.5 - |c_ik|)^2
    d_wall = 0.5 - np.abs(c)
    d_wall = np.maximum(d_wall, 1e-12)
    V += np.sum(1.0 / (d_wall * d_wall))
    
    signs = np.sign(c)
    # grad = 2 * sign(c) / d_wall^3
    grad_wall = 2.0 * signs / (d_wall**3)
    grad[:] += grad_wall.flatten()
    
    return V, grad

def _compute_score(centers):
    """Calculate the radius of the packing."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_dist / 2.0, min_wall_dist)
    return r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    best_score = -1.0
    best_centers = None
    
    # Initial grid 5x5x4
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    # 5x5x4 grid
    grid = np.array([(x, y, z) for z in ticks[:4] for y in ticks for x in ticks])
    base_grid = grid.copy()
    
    np.random.seed(42)
    
    # Try a few restarts with different jitter magnitudes
    jitter_mags = [0.01, 0.02, 0.05, 0.01, 0.02]
    
    for i, mag in enumerate(jitter_mags):
        if time.time() > deadline:
            break
            
        # Jitter
        x0 = base_grid + np.random.uniform(-mag, mag, size=(N, 3))
        # Ensure within bounds
        x0 = np.clip(x0, -0.49, 0.49)
        x0_flat = x0.flatten()
        
        bounds = [(-0.5, 0.5)] * (3 * N)
        
        try:
            res = minimize(_potential_and_grad, x0_flat, jac=True, method='L-BFGS-B', 
                           bounds=bounds, options={'maxiter': 2000, 'maxfun': 10000, 
                                                   'ftol': 1e-10, 'gtol': 1e-8, 'maxstep': 0.1})
        except Exception:
            continue
            
        centers = res.x.reshape(N, 3)
        # Clip to be safe
        centers = np.clip(centers, -0.5, 0.5)
        score = _compute_score(centers)
        
        if score > best_score:
            best_score = score
            best_centers = centers
            
        # Early stop if we hit a very high score?
        # Best known is ~0.111. If we get > 0.110, it's likely near optimal.
        if best_score > 0.110:
            # We can break or continue. Let's continue to try and squeeze more.
            pass

    if best_centers is None:
        return base_grid
    
    return best_centers
# EVOLVE-BLOCK-END
