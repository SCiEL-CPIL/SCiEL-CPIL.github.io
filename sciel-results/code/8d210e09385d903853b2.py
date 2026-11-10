import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # Objective function: Sum of squared violations
    def energy(x, R):
        c = x.reshape(N, 3)
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Pairwise violations
        pair_viol = np.maximum(0, 2 * R - dists)
        
        # Wall violations
        # Distance to nearest wall for each sphere
        # min(0.5 - |x|, 0.5 - |y|, 0.5 - |z|) is the distance to the boundary
        # Violation if dist < R => |c_ik| > 0.5 - R
        abs_c = np.abs(c)
        # Max coordinate absolute value
        max_coord = np.max(abs_c, axis=1)
        wall_viol = np.maximum(0, max_coord - (0.5 - R))
        
        return np.sum(pair_viol**2) + np.sum(wall_viol**2)

    # Gradient is not easily computed analytically for the max/clip operations 
    # but L-BFGS-B can approximate it or we can use finite differences.
    # However, for speed, we can just let L-BFGS-B use its own approximation or 
    # provide a numerical gradient.
    # Actually, the function is differentiable almost everywhere.
    # Let's stick to L-BFGS-B with numerical gradient.
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Initial guess: Jittered grid
    # 5x5x5 grid = 125 points. We take 100.
    # To make it symmetric and good, we can just take a 5x5x5 grid and randomly remove 25.
    # Or better, start with a 4x5x5 grid (100 points) and jitter.
    # 4x5x5 grid:
    # x: 4 points, y: 5 points, z: 5 points.
    # Spacing: x: 1/4=0.25, y: 0.2, z: 0.2.
    # Radius limited by 0.1.
    
    def make_initial_guess(seed):
        rng = np.random.RandomState(seed)
        # 4x5x5 grid
        xs = (np.arange(4) + 0.5) / 4.0 - 0.5
        ys = (np.arange(5) + 0.5) / 5.0 - 0.5
        zs = (np.arange(5) + 0.5) / 5.0 - 0.5
        
        grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
        
        # Add small jitter to break symmetry
        jitter = rng.uniform(-0.01, 0.01, size=(N, 3))
        grid = grid + jitter
        
        # Clip to bounds to ensure validity
        grid = np.clip(grid, -0.5, 0.5)
        return grid.flatten()

    best_r = 0.0
    best_c = None
    
    # Try a few different initial seeds and radius increments
    # We will try to increase R from a low start.
    
    # Strategy: 
    # 1. Start with R=0.08
    # 2. Optimize
    # 3. If success (energy < 1e-5), R += 0.001
    # 4. Repeat until failure or time out.
    
    # To be robust, we might want to restart from scratch if we get stuck.
    # But incremental R usually works well if the initial guess is good.
    
    # Let's try 3 different initializations (seeds 1, 2, 3) and pick the best.
    
    for seed in [1, 2, 3]:
        if time.time() - start_time > time_limit - 10:
            break
            
        x0 = make_initial_guess(seed)
        
        # Start with a conservative radius
        R = 0.08
        # Max radius to try
        max_R = 0.12
        
        while R < max_R:
            if time.time() - start_time > time_limit - 5:
                break
                
            # Optimize for current R
            # Use L-BFGS-B
            # We don't need high precision for the check, just low energy.
            res = minimize(
                lambda x: energy(x, R), 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'maxfev': 1000, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            
            if res.success or res.fun < 1e-6:
                # Check if it's actually valid
                c_test = res.x.reshape(N, 3)
                # Verify
                diff = c_test[:, np.newaxis, :] - c_test[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)
                np.fill_diagonal(dists, np.inf)
                min_pair = np.min(dists)
                wall_dists = 0.5 - np.max(np.abs(c_test), axis=1)
                min_wall = np.min(wall_dists)
                actual_r = min(min_pair / 2, min_wall)
                
                if actual_r >= R - 1e-5:
                    # Accept this configuration and radius
                    x0 = res.x
                    R += 0.001
                else:
                    # Stuck, break
                    break
            else:
                # Optimization failed or energy too high
                break
        
        # After finishing the loop for this seed, record the best
        # Note: x0 holds the last successful position
        c_final = x0.reshape(N, 3)
        
        # Calculate final radius
        diff = c_final[:, np.newaxis, :] - c_final[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        wall_dists = 0.5 - np.max(np.abs(c_final), axis=1)
        min_wall = np.min(wall_dists)
        r_final = min(min_pair / 2, min_wall)
        
        if best_c is None or r_final > best_r:
            best_r = r_final
            best_c = c_final

    if best_c is None:
        # Fallback to simple grid
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]
        
    return best_c
