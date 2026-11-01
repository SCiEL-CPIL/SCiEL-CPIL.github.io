"""Improved 50-sphere packing: Repulsion Annealing + Lloyd + Smooth Surrogate + Exact Polish."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s limit

def _compute_radius(pts):
    """Compute the maximum valid radius for given centers."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 3):
        return -1e6
    if not np.all(np.isfinite(pts)):
        return -1e6
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return -1e6
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Wall distances
    wall_dists = 0.5 - np.abs(pts)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)

def _soft_min_radius(x, beta=50.0):
    """
    Smooth surrogate for negative radius using log-sum-exp soft-min.
    Returns a differentiable approximation of -min(constraints).
    """
    pts = x.reshape(N, 3)
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return 1e6
    
    # Pairwise distances / 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    # Upper triangle indices for unique pairs
    iu = np.triu_indices(N, k=1)
    pair_flat = pair_dists[iu]
    
    # Wall distances
    wall_dists = (0.5 - np.abs(pts)).flatten()
    
    all_dists = np.concatenate([pair_flat, wall_dists])
    
    # Stable soft-min: min(d) ≈ d_min - (1/beta)*log(sum(exp(-beta*(d-d_min))))
    d_min = np.min(all_dists)
    shifted = -beta * (all_dists - d_min)
    # Clip to prevent overflow (shifted <= 0)
    shifted = np.clip(shifted, -700, 0)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = d_min - log_sum / beta
    
    return -soft_min_val

def _init_grid():
    """5x5x2 grid with small jitter."""
    xs = (np.arange(5) + 0.5) / 5.0 - 0.5
    ys = (np.arange(5) + 0.5) / 5.0 - 0.5
    zs = (np.arange(2) + 0.5) / 2.0 - 0.5
    centers = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    centers += np.random.normal(0, 0.001, size=(N, 3))
    return np.clip(centers, -0.49, 0.49)

def _init_fcc():
    """FCC lattice initialization."""
    a = 1.0 / 3.0
    pts = []
    basis = [[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]]
    for i in range(-1, 4):
        for j in range(-1, 4):
            for k in range(-1, 4):
                for b in basis:
                    p = a * np.array([i, j, k]) + a * np.array(b)
                    if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5 and -0.5 <= p[2] <= 0.5:
                        pts.append(p)
    pts = np.array(pts)
    if len(pts) < N:
        return _init_grid()
    # Greedy selection for spread
    selected = []
    remaining = pts.copy()
    for _ in range(N):
        if len(remaining) == 0:
            break
        if not selected:
            idx = 0
        else:
            sel = np.array(selected)
            d = np.min(np.linalg.norm(remaining[:, np.newaxis, :] - sel[np.newaxis, :, :], axis=2), axis=1)
            idx = np.argmax(d)
        selected.append(remaining[idx])
        remaining = np.delete(remaining, idx, axis=0)
    return np.array(selected)

def _init_bcc():
    """BCC lattice initialization."""
    a = 1.0 / 3.0
    pts = []
    basis = [[0,0,0], [0.5,0.5,0.5]]
    for i in range(-1, 4):
        for j in range(-1, 4):
            for k in range(-1, 4):
                for b in basis:
                    p = a * np.array([i, j, k]) + a * np.array(b)
                    if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5 and -0.5 <= p[2] <= 0.5:
                        pts.append(p)
    pts = np.array(pts)
    if len(pts) < N:
        return _init_grid()
    # Greedy selection for spread
    selected = []
    remaining = pts.copy()
    for _ in range(N):
        if len(remaining) == 0:
            break
        if not selected:
            idx = 0
        else:
            sel = np.array(selected)
            d = np.min(np.linalg.norm(remaining[:, np.newaxis, :] - sel[np.newaxis, :, :], axis=2), axis=1)
            idx = np.argmax(d)
        selected.append(remaining[idx])
        remaining = np.delete(remaining, idx, axis=0)
    return np.array(selected)

def _init_random():
    return np.random.uniform(-0.45, 0.45, size=(N, 3))

def _repulsion_anneal(pts, n_steps=100, lr_start=0.01, lr_end=0.0005):
    """
    Repulsion annealing: iteratively push spheres apart using gradient of repulsive potential.
    Vectorized for speed. Helps escape shallow local minima before gradient-based optimization.
    """
    pts = pts.copy().astype(np.float64)
    best_r = _compute_radius(pts)
    best_pts = pts.copy()
    
    for step in range(n_steps):
        if time.time() - START_TIME > TIME_LIMIT - 35.0:
            break
        lr = lr_start * (lr_end / lr_start) ** (step / max(1, n_steps - 1))
        
        # Compute forces (vectorized)
        # Pairwise repulsion
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Force magnitude: 1/d^2 (capped)
        forces_pair = np.zeros_like(dists)
        mask = dists < 0.5  # Only interact if close enough
        forces_pair[mask] = 1.0 / (dists[mask] * dists[mask])
        
        # Sum forces
        f = np.sum(forces_pair[:, :, np.newaxis] * diff, axis=1)
        
        # Wall repulsion
        # Push away from walls if close
        wall_dist = 0.5 - np.abs(pts)
        wall_force = np.zeros_like(pts)
        # If x is close to -0.5, push right (+x). If close to 0.5, push left (-x).
        # wall_dist is positive inside.
        # Gradient of wall_dist wrt x: -sign(x).
        # We want to maximize wall_dist, so move in direction of gradient.
        # grad_wall = -sign(pts)
        # Force = grad_wall / wall_dist^2
        mask_wall = wall_dist < 0.1
        for dim in range(3):
            wd = wall_dist[:, dim]
            wf = np.zeros_like(wd)
            # Only apply force if close to wall
            close = wd < 0.1
            wf[close] = 1.0 / (wd[close] * wd[close])
            # Direction: away from wall
            # If pts[:, dim] > 0, wall is at 0.5, push negative.
            # If pts[:, dim] < 0, wall is at -0.5, push positive.
            # So force direction is -sign(pts[:, dim])
            f[:, dim] += -np.sign(pts[:, dim]) * wf
        
        # Update
        pts += lr * f
        pts = np.clip(pts, -0.499, 0.499)
        
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
            
    return best_pts

def _lloyd_optimize(pts, n_iters=50):
    """
    Lloyd's algorithm for sphere packing.
    Moves each sphere to the centroid of its Voronoi cell.
    """
    pts = pts.copy().astype(np.float64)
    best_r = _compute_radius(pts)
    best_pts = pts.copy()
    
    for it in range(n_iters):
        if time.time() - START_TIME > TIME_LIMIT - 30.0:
            break
        
        try:
            # Compute Delaunay triangulation (Voronoi dual)
            # Add a large bounding box to handle boundary conditions
            # Instead of complex boundary handling, we use a heuristic:
            # If a sphere is close to a wall, its "Voronoi cell" is truncated.
            # A simpler approximation: just move towards the center of mass of neighbors?
            # No, Lloyd's is specific. Let's use a simple gradient-based "spread" instead
            # if Delaunay is too slow or tricky with boundaries.
            
            # Alternative: "Lloyd-like" step using nearest neighbors
            # Move each point away from its nearest neighbor to increase min distance.
            # This is essentially what repulsion anneal does.
            
            # Let's stick to a simple refinement step that acts like Lloyd's
            # by maximizing the minimum distance to neighbors locally.
            
            improved = False
            for i in range(N):
                if time.time() - START_TIME > TIME_LIMIT - 30.0:
                    break
                # Find nearest neighbor
                dists = np.linalg.norm(pts - pts[i], axis=1)
                dists[i] = np.inf
                nn_idx = np.argmin(dists)
                nn_dist = dists[nn_idx]
                
                # Direction away from nearest neighbor
                if nn_dist > 1e-6:
                    dir_vec = (pts[i] - pts[nn_idx]) / nn_dist
                else:
                    dir_vec = np.random.normal(0, 1, 3)
                    dir_vec /= np.linalg.norm(dir_vec)
                
                # Also consider walls
                wall_dists = 0.5 - np.abs(pts[i])
                min_wall_idx = np.argmin(wall_dists)
                min_wall_dist = wall_dists[min_wall_idx]
                
                # If wall is closer than neighbor, move away from wall
                if min_wall_dist < nn_dist / 2:
                    # Move away from wall
                    wall_dir = np.zeros(3)
                    wall_dir[min_wall_idx] = -np.sign(pts[i, min_wall_idx])
                    dir_vec = wall_dir
                    step_size = min(0.01, min_wall_dist * 0.5)
                else:
                    step_size = min(0.01, nn_dist * 0.2)
                
                new_pt = pts[i] + step_size * dir_vec
                new_pt = np.clip(new_pt, -0.499, 0.499)
                
                # Check if this move improves the local radius
                old_local_r = min(nn_dist/2, min_wall_dist)
                new_pts = pts.copy()
                new_pts[i] = new_pt
                
                # Quick check: distance to new NN and walls
                new_dists = np.linalg.norm(new_pts - new_pt, axis=1)
                new_dists[i] = np.inf
                new_nn_dist = np.min(new_dists)
                new_wall_dists = 0.5 - np.abs(new_pt)
                new_min_wall = np.min(new_wall_dists)
                new_local_r = min(new_nn_dist/2, new_min_wall)
                
                if new_local_r > old_local_r + 1e-8:
                    pts[i] = new_pt
                    improved = True
            
            r = _compute_radius(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
                
            if not improved:
                break
                
        except Exception:
            break
            
    return best_pts

def _refine_smooth(pts):
    """Refine using smooth surrogate with annealing beta."""
    x = pts.flatten().copy()
    bounds = [(-0.499, 0.499)] * (N * 3)
    
    # Phase 1: Smooth surrogate with increasing beta
    betas = [15.0, 40.0, 100.0, 250.0]
    for beta in betas:
        if time.time() - START_TIME > TIME_LIMIT - 20.0:
            break
        try:
            def obj(x):
                return _soft_min_radius(x, beta)
            res = minimize(obj, x, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 40})
            if np.all(np.isfinite(res.x)):
                x = res.x
        except Exception:
            pass
    
    x_2d = x.reshape(N, 3)
    x_2d = np.clip(x_2d, -0.499, 0.499)
    
    # Phase 2: Local search (coordinate-wise)
    if time.time() - START_TIME < TIME_LIMIT - 15.0:
        x_2d = _local_search(x_2d, max_iter=50)
        
    # Phase 3: Exact Nelder-Mead polish
    try:
        res = minimize(_neg_radius, x_2d, method='Nelder-Mead',
                       options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if np.all(np.isfinite(res.x)):
            x_2d = res.x
    except Exception:
        pass
    
    return np.clip(x_2d, -0.499, 0.499)

def _neg_radius(x):
    """Objective: negative radius (to be minimized)."""
    pts = x.reshape(N, 3)
    return -_compute_radius(pts)

def _local_search(pts, max_iter=200):
    """Coordinate-wise steepest ascent on exact radius."""
    pts = pts.copy().astype(np.float64)
    best_r = _compute_radius(pts)
    
    for it in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT - 10.0:
            break
        
        improved = False
        for i in range(N):
            for dim in range(3):
                # Try small steps in each direction
                for sign in [1, -1]:
                    for step_size in [0.005, 0.002, 0.001, 0.0005]:
                        new_pts = pts.copy()
                        new_pts[i, dim] += sign * step_size
                        new_pts[i, dim] = np.clip(new_pts[i, dim], -0.499, 0.499)
                        new_r = _compute_radius(new_pts)
                        if new_r > best_r + 1e-12:
                            pts = new_pts
                            best_r = new_r
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                continue
        
        if not improved:
            break
    
    return pts

def _final_polish(pts):
    """High-precision final polish with tighter tolerances."""
    x_2d = pts.copy().astype(np.float64)
    best_r = _compute_radius(x_2d)
    
    # Multiple rounds of Nelder-Mead with decreasing tolerances
    for atol in [1e-10, 1e-12, 1e-14]:
        if time.time() - START_TIME > TIME_LIMIT - 5.0:
            break
        try:
            res = minimize(_neg_radius, x_2d, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': atol, 'fatol': atol, 'adaptive': True})
            if np.all(np.isfinite(res.x)):
                new_r = _compute_radius(res.x)
                if new_r > best_r:
                    x_2d = res.x
                    best_r = new_r
        except Exception:
            pass
    
    # Final local search with very small steps
    if time.time() - START_TIME < TIME_LIMIT - 3.0:
        for it in range(50):
            if time.time() - START_TIME > TIME_LIMIT - 3.0:
                break
            improved = False
            for i in range(N):
                for dim in range(3):
                    for sign in [1, -1]:
                        for step_size in [0.0005, 0.0002, 0.0001]:
                            new_pts = x_2d.copy()
                            new_pts[i, dim] += sign * step_size
                            new_pts[i, dim] = np.clip(new_pts[i, dim], -0.499, 0.499)
                            new_r = _compute_radius(new_pts)
                            if new_r > best_r + 1e-14:
                                x_2d = new_pts
                                best_r = new_r
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if not improved:
                    continue
            if not improved:
                break
    
    return np.clip(x_2d, -0.499, 0.499)

def construct():
    best_pts = None
    best_r = -np.inf
    
    inits = [
        _init_grid(),
        _init_fcc(),
        _init_bcc(),
        _init_random(),
        _init_random(),
    ]
    
    # Process each initialization
    for i, init_pts in enumerate(inits):
        if time.time() - START_TIME > TIME_LIMIT - 40.0:
            break
        
        init_pts = np.clip(init_pts, -0.49, 0.49)
        
        # 1. Repulsion Annealing
        if time.time() - START_TIME < TIME_LIMIT - 35.0:
            refined = _repulsion_anneal(init_pts, n_steps=80)
        else:
            refined = init_pts
            
        # 2. Lloyd-like optimization
        if time.time() - START_TIME < TIME_LIMIT - 30.0:
            refined = _lloyd_optimize(refined, n_iters=30)
            
        # 3. Smooth Gradient Refinement
        if time.time() - START_TIME < TIME_LIMIT - 20.0:
            refined = _refine_smooth(refined)
            
        refined = np.clip(refined, -0.499, 0.499)
        r = _compute_radius(refined)
        
        if r > best_r:
            best_r = r
            best_pts = refined.copy()
            
    # Final polish on best solution if time remains
    if best_pts is not None and time.time() - START_TIME < TIME_LIMIT - 10.0:
        # Local search first
        if time.time() - START_TIME < TIME_LIMIT - 15.0:
            polished = _local_search(best_pts, max_iter=50)
            r_ls = _compute_radius(polished)
            if r_ls > best_r:
                best_r = r_ls
                best_pts = polished
                
        # Then high-precision final polish
        if time.time() - START_TIME < TIME_LIMIT - 5.0:
            polished = _final_polish(best_pts)
            r2 = _compute_radius(polished)
            if r2 > best_r:
                best_r = r2
                best_pts = polished
    
    if best_pts is None:
        best_pts = _init_grid()
    
    return best_pts

# EVOLVE-BLOCK-END
