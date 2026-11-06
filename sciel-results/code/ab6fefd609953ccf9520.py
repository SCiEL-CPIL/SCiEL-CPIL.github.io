"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 115.0  # safety margin before 120s kill

def _compute_radius(c):
    """Compute the achievable common radius for centres c (N,3) vectorized."""
    n = c.shape[0]
    # Wall distances
    wall = 0.5 - np.max(np.abs(c), axis=1)
    min_wall = np.min(wall)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    return min(min_dist / 2.0, min_wall)

def _relax_spheres(c, r, max_iter=1000, tol=1e-6):
    """
    Relax spheres to satisfy radius r using vectorized repulsion.
    Returns (c_new, converged)
    """
    c = c.copy()
    n = c.shape[0]
    limit = 0.5 - r
    lr = 0.5  # Learning rate
    
    for it in range(max_iter):
        # 1. Pairwise repulsion
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero
        dist_safe = np.where(dist > 1e-8, dist, 1.0)
        
        # Overlap: positive if dist < 2r
        overlap = np.where(dist < 2*r, 2*r - dist, 0.0)
        
        # Force vector for each pair (i, j) is along diff[i, j]
        # Force on i from j: (c[i] - c[j]) / dist * overlap
        # Total force on i: sum_j (c[i] - c[j]) / dist[i, j] * overlap[i, j]
        
        # term[i, j] = diff[i, j] * overlap[i, j] / dist_safe[i, j]
        # F[i] = sum_j term[i, j]
        
        # Note: overlap is 0 for i=j (dist=0 -> safe=1, overlap=2r, but diff=0 so term=0)
        # So we don't need to mask diagonal explicitly for the sum, but it's 0 anyway.
        
        force = np.sum(diff * (overlap / dist_safe)[:, :, np.newaxis], axis=1)
        
        # 2. Update positions
        c += lr * force
        
        # 3. Project to walls
        c = np.clip(c, -limit, limit)
        
        # 4. Check convergence
        # If max force is small, we are good.
        if np.max(np.abs(force)) < tol:
            return c, True
            
    return c, False

def construct():
    t0 = time.time()
    
    # Initial grid: 5 x 5 x 4 = 100
    nx, ny, nz = 5, 5, 4
    xs = (np.arange(nx) + 0.5) / nx - 0.5
    ys = (np.arange(ny) + 0.5) / ny - 0.5
    zs = (np.arange(nz) + 0.5) / nz - 0.5
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    
    # Add small jitter to break symmetry
    c0 = grid[:N] + np.random.uniform(-0.005, 0.005, (N, 3))
    c0 = np.clip(c0, -0.49, 0.49)
    
    r_current = _compute_radius(c0)
    best_c = c0.copy()
    best_r = r_current
    
    # Try to increase radius incrementally
    r_target = r_current * 1.01
    max_r_target = 0.112  # Slightly above known best 0.11138
    
    restarts = 0
    max_restarts = 20
    
    while time.time() - t0 < _DEADLINE:
        if r_target > max_r_target:
            break
        
        c_curr = best_c.copy()
        
        # Relax
        c_new, converged = _relax_spheres(c_curr, r_target, max_iter=500)
        
        if converged:
            r_actual = _compute_radius(c_new)
            if r_actual >= r_target - 1e-9:
                best_c = c_new
                best_r = r_actual
                r_target *= 1.002
                restarts = 0
            else:
                # Failed to reach target, try smaller increment
                r_target *= 1.0005
                if r_target <= best_r + 1e-9:
                     # Stuck, restart
                     restarts += 1
                     if restarts > max_restarts:
                         break
                     c_curr = np.random.uniform(-0.4, 0.4, (N, 3))
        else:
             # Not converged, restart
             restarts += 1
             if restarts > max_restarts:
                 break
             c_curr = np.random.uniform(-0.4, 0.4, (N, 3))
             # Maybe lower r_target a bit?
             r_target *= 0.99
             
    return best_c
# EVOLVE-BLOCK-END
