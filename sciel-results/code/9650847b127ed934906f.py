"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit

    # 1. Initial Configuration: 5x5x4 grid
    # The cube is [-0.5, 0.5]^3
    # We want 100 points. 5*5*4 = 100.
    # Spacing in x, y is 1/5 = 0.2. Spacing in z is 1/4 = 0.25.
    # Centers should be at (i+0.5)/m - 0.5
    m_x, m_y, m_z = 5, 5, 4
    x = (np.arange(m_x) + 0.5) / m_x - 0.5
    y = (np.arange(m_y) + 0.5) / m_y - 0.5
    z = (np.arange(m_z) + 0.5) / m_z - 0.5
    
    # Create grid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # 2. Optimization
    # We minimize the sum of inverse powers of distances to push points apart.
    # This effectively maximizes the minimum distance.
    
    def objective(flat_points):
        p = flat_points.reshape(-1, 3)
        
        # Check boundaries
        # Distance to walls: 0.5 - abs(coord)
        # If a point is outside, penalty should be huge.
        # For simplicity in the objective, we assume points stay inside due to initial setup and penalty.
        
        # Wall penalties
        # dist_wall = 0.5 - np.abs(p)
        # wall_pen = np.sum(1.0 / (dist_wall**4)) # Dangerous if dist_wall is 0 or negative
        
        # To be safe, let's use a softer wall penalty or ensure we don't cross.
        # Actually, L-BFGS-B can handle bounds. Let's use bounds.
        
        # Pairwise distances
        # Using broadcasting
        # diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        # dist_sq = np.sum(diff**2, axis=-1)
        
        # This is O(N^2) memory and compute. N=100 -> 10000 pairs. Manageable.
        # But we can be smarter or just compute directly.
        
        # Let's compute sum of 1/d^4 for all pairs i<j
        # And sum of 1/(0.5 - |c_i|)^4 for all coords
        
        # Wall term
        # We add a large penalty if points get close to walls
        d_wall = 0.5 - np.abs(p)
        # If d_wall <= 0, it's invalid. We want to keep it positive.
        # To avoid division by zero or negative, we can clip or use a large constant.
        # But with bounds, we shouldn't go outside.
        # Let's use bounds in the optimizer.
        
        # Pairwise term
        # Compute upper triangle indices
        i_idx, j_idx = np.triu_indices(N, k=1)
        diffs = p[i_idx] - p[j_idx]
        dist_sq = np.sum(diffs**2, axis=1)
        
        # Avoid division by zero
        dist_sq = np.maximum(dist_sq, 1e-12)
        
        pair_pen = np.sum(1.0 / (dist_sq**2)) # This is sum 1/d^4
        
        # Wall penalty
        # We want to maximize distance to walls.
        # So we minimize 1/d_wall^4
        d_wall_sq = np.maximum(d_wall**2, 1e-12)
        wall_pen = np.sum(1.0 / (d_wall_sq**2)) # Sum 1/d^4 for all coords
        
        return pair_pen + 100.0 * wall_pen # Weight walls more? Or equally?
        # Actually, the radius is limited by min(pair_dist/2, wall_dist).
        # So we should balance them.
        # If we just sum them, the optimizer will try to balance them naturally.
        
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Initial guess
    x0 = points.ravel()
    
    # Run optimization
    # L-BFGS-B is good for large scale with bounds.
    # We might need to run it multiple times or with different initializations if it gets stuck.
    # Given time constraints, we'll run one long optimization.
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12, 'maxfun': 10000})
        
        if time.time() > deadline:
             # If we exceed deadline, return current best or initial?
             # The minimize function returns best found so far.
             pass
             
        # Check if we have time for a second refinement or different start
        if time.time() < deadline - 10:
            # Try a perturbed start?
            # Or just trust the result.
            pass
            
        best_points = res.x.reshape(-1, 3)
        
    except Exception as e:
        # If optimization fails, return initial grid
        best_points = points

    # Final check: ensure points are within bounds (clip just in case)
    best_points = np.clip(best_points, -0.5, 0.5)
    
    return best_points

# EVOLVE-BLOCK-END
