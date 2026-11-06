"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 115.0  # safety margin before 120s kill

def _compute_radius(c):
    """Compute the achievable common radius for centres c (N,3)."""
    n = c.shape[0]
    # Wall distances
    wall = 0.5 - np.max(np.abs(c), axis=1)
    min_wall = np.min(wall)
    # Pairwise distances - vectorized for 100 points (100x100x3 is fine)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    return min(min_dist / 2.0, min_wall)

def _relax_spheres(c, r, max_iter=2000, tol=1e-8):
    """
    Relax spheres to satisfy radius r using vectorized repulsion.
    Returns (c_new, converged)
    """
    c = c.copy()
    n = c.shape[0]
    limit = 0.5 - r
    # Adaptive learning rate
    lr = 0.5
    
    for it in range(max_iter):
        # Pairwise differences and distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        
        # Avoid division by zero
        dist_safe = np.where(dist > 1e-8, dist, 1.0)
        
        # Overlap: positive if dist < 2r
        overlap = np.where(dist < 2*r, 2*r - dist, 0.0)
        
        # Force on each sphere from all others
        # F[i] = sum_j (c[i] - c[j]) / dist[i,j] * overlap[i,j]
        force = np.sum(diff * (overlap / dist_safe)[:, :, np.newaxis], axis=1)
        
        # Update positions
        c += lr * force
        
        # Project to walls
        c = np.clip(c, -limit, limit)
        
        # Check convergence
        max_force = np.max(np.abs(force))
        if max_force < tol:
            return c, True
            
    return c, False

def _make_grid(nx, ny, nz, jitter=0.0):
    """Create a grid of points in the unit cube centered at origin."""
    xs = (np.arange(nx) + 0.5) / nx - 0.5
    ys = (np.arange(ny) + 0.5) / ny - 0.5
    zs = (np.arange(nz) + 0.5) / nz - 0.5
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    if jitter > 0:
        grid = grid + np.random.uniform(-jitter, jitter, grid.shape)
    return grid

def construct():
    t0 = time.time()
    rng = np.random.RandomState(42)
    
    best_c = None
    best_r = 0.0
    
    # Try multiple grid configurations and optimizations
    grid_configs = [
        (5, 5, 4),  # 100 points
        (4, 5, 5),  # 100 points
        (5, 4, 5),  # 100 points
    ]
    
    for nx, ny, nz in grid_configs:
        if time.time() - t0 > _DEADLINE:
            break
        
        for jitter in [0.0, 0.002, 0.005, 0.01]:
            if time.time() - t0 > _DEADLINE:
                break
            
            c0 = _make_grid(nx, ny, nz, jitter=jitter)
            c0 = np.clip(c0, -0.49, 0.49)
            
            r_current = _compute_radius(c0)
            if best_r is None or r_current > best_r:
                best_r = r_current
                best_c = c0.copy()
            
            # Incrementally increase radius
            r_target = r_current * 1.005
            max_r_target = 0.112  # Slightly above known best 0.11138
            
            while time.time() - t0 < _DEADLINE and r_target <= max_r_target:
                c_test, converged = _relax_spheres(c0, r_target, max_iter=1500)
                
                if converged:
                    r_actual = _compute_radius(c_test)
                    if r_actual >= r_target - 1e-9 and r_actual > best_r:
                        best_r = r_actual
                        best_c = c_test.copy()
                        c0 = c_test.copy()  # Use improved config for next iteration
                        r_target *= 1.005
                    else:
                        # Failed to achieve target, try smaller increment
                        r_target = r_current * 1.001 + 0.0001
                else:
                    # Not converged, try smaller target
                    r_target = (r_target + r_current) / 2.0
                    if r_target <= r_current + 1e-9:
                        break
            
            # After grid optimization, try random perturbations
            if best_c is not None and time.time() - t0 < _DEADLINE:
                for _ in range(5):
                    if time.time() - t0 > _DEADLINE:
                        break
                    c_pert = best_c + rng.randn(*best_c.shape) * 0.01
                    c_pert = np.clip(c_pert, -(0.5 - best_r), 0.5 - best_r)
                    r_pert = _compute_radius(c_pert)
                    if r_pert > best_r:
                        best_r = r_pert
                        best_c = c_pert.copy()
    
    return best_c
# EVOLVE-BLOCK-END
