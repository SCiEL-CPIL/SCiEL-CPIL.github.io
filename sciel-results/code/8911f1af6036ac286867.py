"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s (evaluator kills at 150s)
    
    # 1. Generate a good initial configuration using a jittered grid
    def make_initial_config():
        # Use a 5x5x4 grid = 100 points
        # This gives a good starting point
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks[:4] for y in ticks for x in ticks])
        # Add small random jitter to break symmetry
        np.random.seed(42)
        jitter = 0.01 * np.random.randn(*grid.shape)
        grid = grid + jitter
        # Clamp to valid range
        grid = np.clip(grid, -0.45, 0.45)
        return grid
    
    def score(x):
        """Compute the score: min of all pairwise distances/2 and distances to walls"""
        x = x.reshape(-1, 3)
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        # Distance to walls
        wall_dists = 0.5 - np.abs(x)
        min_wall = np.min(wall_dists)
        return min(min_pair, min_wall)
    
    def objective(x):
        """Negative score for minimization"""
        return -score(x)
    
    def objective_smooth(x, beta=50.0):
        """Smooth approximation using soft-min"""
        x = x.reshape(-1, 3)
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        # Distance to walls
        wall_dists = 0.5 - np.abs(x)
        min_wall = np.min(wall_dists)
        
        # Soft min using log-sum-exp trick
        vals = np.array([min_pair, min_wall])
        max_val = np.max(vals)
        soft_min = max_val - np.log(np.sum(np.exp(vals - max_val)))
        return -soft_min
    
    # Start with initial config
    x0 = make_initial_config()
    best_x = x0.copy()
    best_score = score(x0)
    
    # Iterative optimization with gradient-free methods
    # Use Nelder-Mead or Powell for robustness
    methods = ['Nelder-Mead', 'Powell']
    
    for method in methods:
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            
            # Restart from best or add small perturbation
            if trial == 0:
                x_start = best_x.copy()
            else:
                x_start = best_x + 0.005 * np.random.randn(*best_x.shape)
                x_start = np.clip(x_start, -0.49, 0.49)
            
            # Use bounds to keep points inside cube
            bounds = [(-0.5, 0.5)] * (N * 3)
            
            try:
                if method == 'Nelder-Mead':
                    res = minimize(objective, x_start, method='Nelder-Mead', 
                                  options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
                else:
                    res = minimize(objective, x_start, method='Powell', 
                                  bounds=bounds, 
                                  options={'maxiter': 300, 'xtol': 1e-6, 'ftol': 1e-8})
                
                if res.fun < -best_score:
                    best_score = -res.fun
                    best_x = res.x.reshape(-1, 3)
            except Exception:
                pass
        
        if time.time() > deadline - 10:
            break
    
    # Final polish: local refinement
    if time.time() < deadline - 5:
        for polish_iter in range(50):
            if time.time() > deadline - 2:
                break
            
            # Try to improve by moving each point slightly
            improved = False
            for i in range(N):
                if time.time() > deadline - 2:
                    break
                    
                # Compute current constraints for point i
                # Find nearest neighbor and nearest wall
                diff = best_x[i] - best_x
                dists = np.linalg.norm(diff, axis=1)
                dists[i] = np.inf
                nearest_j = np.argmin(dists)
                min_pair_dist = dists[nearest_j] / 2.0
                
                wall_dists = 0.5 - np.abs(best_x[i])
                nearest_wall = np.argmin(wall_dists)
                min_wall_dist = wall_dists[nearest_wall]
                
                current_r = min(min_pair_dist, min_wall_dist)
                
                # Try small moves in each direction
                for dim in range(3):
                    for direction in [-1, 1]:
                        new_x = best_x.copy()
                        new_x[i, dim] += direction * 1e-4
                        new_x[i, dim] = np.clip(new_x[i, dim], -0.499, 0.499)
                        
                        # Quick check if this improves
                        new_score = score(new_x)
                        if new_score > best_score + 1e-10:
                            best_x = new_x
                            best_score = new_score
                            improved = True
                            break
                    if improved:
                        break
            if not improved:
                break
    
    # Ensure valid output
    best_x = np.clip(best_x, -0.4999, 0.4999)
    
    return best_x
# EVOLVE-BLOCK-END
