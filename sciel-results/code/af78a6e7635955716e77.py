import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _radius(pts):
    """
    Calculate the maximum common radius for the given centers.
    pts: (N, 3) array of centers.
    """
    # Pairwise distances
    # diff[i, j, k] = pts[i, k] - pts[j, k]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Wall distances
    # Distance to closest wall
    # Cube is [-0.5, 0.5]
    # dist to wall = min(x - (-0.5), 0.5 - x) = min(x+0.5, 0.5-x) = 0.5 - |x|
    wall_dists = 0.5 - np.abs(pts)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _init_grid():
    """
    Initialize points on a jittered 4x4x4 grid.
    """
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Shuffle and take N
    idx = np.random.choice(len(grid), N, replace=False)
    pts = grid[idx]
    # Jitter
    pts += np.random.normal(0, 0.02, (N, 3))
    # Clip to cube
    pts = np.clip(pts, -0.5, 0.5)
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s buffer for safety
    
    best_pts = None
    best_r = 0.0
    
    # Number of restarts
    num_restarts = 4
    time_per_restart = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # 1. Initialization
        pts = _init_grid()
        r = _radius(pts)
        
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
            
        # 2. Simulated Annealing
        T = 0.005
        T_min = 1e-5
        cooling = 0.9995
        step_size = 0.05
        no_improve_count = 0
        last_improve_time = time.time()
        
        while time.time() < start_time + time_per_restart:
            # Perturb a random point
            i = np.random.randint(N)
            delta = np.random.normal(0, step_size, 3)
            new_pts = pts.copy()
            new_pts[i] += delta
            
            # Optional: Clip to cube to avoid huge penalties? 
            # The radius function handles it, but clipping keeps numbers sane.
            new_pts[i] = np.clip(new_pts[i], -0.5, 0.5)
            
            new_r = _radius(new_pts)
            
            # Acceptance
            if new_r > r or np.random.random() < np.exp((new_r - r) / T):
                r = new_r
                pts = new_pts
                if r > best_r:
                    best_r = r
                    best_pts = pts.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
            # Adaptive Step Size
            if no_improve_count > 100:
                step_size *= 0.9
                no_improve_count = 0
                
            # Cooling
            T *= cooling
            
            # If T is very low, break
            if T < T_min:
                break

        # 3. Local Polish (Greedy Hill Climbing)
        # Run for a short fixed time or step count
        polish_deadline = time.time() + 1.0 # 1 second polish
        polish_step = 0.01
        while time.time() < polish_deadline:
            i = np.random.randint(N)
            delta = np.random.normal(0, polish_step, 3)
            new_pts = pts.copy()
            new_pts[i] += delta
            new_pts[i] = np.clip(new_pts[i], -0.5, 0.5)
            
            new_r = _radius(new_pts)
            
            if new_r > r:
                r = new_r
                pts = new_pts
                if r > best_r:
                    best_r = r
                    best_pts = pts.copy()
            else:
                # If stuck, reduce step
                if np.random.random() < 0.01:
                    polish_step *= 0.95

    # Final check: Ensure best_pts is valid
    if best_pts is None:
        # Fallback to seed-like grid if something went wrong
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pts = grid[:N]
        
    return best_pts
