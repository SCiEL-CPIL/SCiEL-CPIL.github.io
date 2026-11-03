import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin below 150s limit
    
    def calc_radius(centers):
        """Calculate the common radius for given centers."""
        # Minimum distance between any two centers / 2
        dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_center_dist = np.min(dists)
        
        # Minimum distance to walls
        wall_dists = 0.5 - np.abs(centers)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_center_dist / 2.0, min_wall_dist)
    
    def objective(x):
        """Negative radius to be minimized."""
        centers = x.reshape(N, 3)
        return -calc_radius(centers)
    
    def smooth_objective(x):
        """
        Smooth approximation: minimize a soft version that penalizes all close pairs.
        Uses sum of exp(-k * dist) for pairs, which is differentiable.
        """
        centers = x.reshape(N, 3)
        # Pairwise distances
        dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(centers)
        min_wall_dist = np.min(wall_dists)
        
        # Soft penalty for close pairs: sum of exp(-lambda * dist)
        # Large lambda makes it sharp (approximates min)
        # We want to maximize min dist, so minimize sum of exp(-lambda * dist)
        # Actually, to maximize min, we can minimize max of (target - dist)
        # Let's use a simpler approach: minimize sum of 1/dist for pairs within a threshold
        # Or use the "max-min" via smooth max: -log(sum(exp(-dist/T))) / log? No.
        
        # Let's use: minimize sum of max(0, r_target - dist)^2 + max(0, r_target - wall_dist)^2
        # This is a feasibility approach.
        
        # Better smooth objective for optimization:
        # Minimize: -min_dist is non-differentiable.
        # Use: sum of exp(-dist / T) is high when dist is small. Minimizing this pushes dist up.
        T = 0.05
        pair_penalty = np.sum(np.exp(-dists / T))
        
        # Wall penalty
        wall_penalty = np.sum(np.exp(-wall_dists.flatten() / T))
        
        return pair_penalty + 10 * wall_penalty

    # Initialize with a good starting point
    # Use a 4x4x4 grid (64 points), randomly pick 50, then optimize
    rng = np.random.default_rng(42)
    
    best_centers = None
    best_radius = -np.inf
    
    # Multiple restarts
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Different initialization strategies
        if restart == 0:
            # 4x4x4 grid, drop 14 randomly
            m = 4
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            indices = rng.choice(len(grid), size=N, replace=False)
            centers = grid[indices].copy()
        elif restart == 1:
            # 5x5x5 grid, drop 75 randomly
            m = 5
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            indices = rng.choice(len(grid), size=N, replace=False)
            centers = grid[indices].copy()
        elif restart == 2:
            # Random points in cube
            centers = rng.uniform(-0.45, 0.45, size=(N, 3))
        elif restart == 3:
            # 3x4x5 grid = 60 points, drop 10
            # x: 3, y: 4, z: 5
            tx = (np.arange(3) + 0.5) / 3 - 0.5
            ty = (np.arange(4) + 0.5) / 4 - 0.5
            tz = (np.arange(5) + 0.5) / 5 - 0.5
            grid = np.array([(x, y, z) for z in tz for y in ty for x in tx])
            indices = rng.choice(len(grid), size=N, replace=False)
            centers = grid[indices].copy()
        elif restart == 4:
            # 2x5x5 = 50 points exactly
            tx = (np.arange(2) + 0.5) / 2 - 0.5
            ty = (np.arange(5) + 0.5) / 5 - 0.5
            tz = (np.arange(5) + 0.5) / 5 - 0.5
            grid = np.array([(x, y, z) for z in tz for y in ty for x in tx])
            centers = grid.copy()
        else:
            # Perturbed version of previous best
            if best_centers is not None:
                centers = best_centers + rng.normal(0, 0.05, size=(N, 3))
                centers = np.clip(centers, -0.49, 0.49)
            else:
                centers = rng.uniform(-0.45, 0.45, size=(N, 3))
        
        x0 = centers.flatten()
        
        # Stage 1: Smooth optimization
        # Use L-BFGS-B with smooth objective
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res_smooth = minimize(
                smooth_objective, x0, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if time.time() > deadline - 5:
                break
        except Exception:
            pass
        
        x_opt = res_smooth.x if 'res_smooth' in dir() else x0
        
        # Stage 2: Direct optimization of negative radius (non-smooth but works well from good start)
        # Use Nelder-Mead or BFGS on the non-smooth objective
        # Since it's non-differentiable, use Nelder-Mead or COBYLA
        try:
            res_direct = minimize(
                objective, x_opt, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if time.time() > deadline - 5:
                break
        except Exception:
            pass
        
        x_final = res_direct.x if 'res_direct' in dir() else x_opt
        
        # Stage 3: Fine-tuning with smaller steps
        try:
            res_fine = minimize(
                objective, x_final, method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'initial_simplex_size': 0.01}
            )
            if time.time() > deadline - 5:
                break
        except Exception:
            pass
        
        x_best = res_fine.x if 'res_fine' in dir() else x_final
        
        # Ensure points are strictly inside the cube
        centers_final = x_best.reshape(N, 3)
        centers_final = np.clip(centers_final, -0.499999, 0.499999)
        
        radius = calc_radius(centers_final)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers_final.copy()
    
    # Final check and return
    if best_centers is None:
        # Fallback: simple grid
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers
