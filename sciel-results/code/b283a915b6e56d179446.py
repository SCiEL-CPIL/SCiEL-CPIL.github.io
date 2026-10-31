import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Objective function to minimize: -sum(radii) + penalties
    # Variables: x (N), y (N), r (N)
    # Total dimensions: 3*N
    
    def objective(params):
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:]
        
        # 1. Maximize sum of radii (minimize negative sum)
        val = -np.sum(r)
        
        # 2. Penalty for overlaps
        # For each pair (i, j), ensure dist(i, j) >= r_i + r_j
        # Penalty: sum( max(0, r_i + r_j - dist(i, j))^2 )
        overlap_penalty = 0.0
        # Vectorized computation for pairs is complex, loop is fine for N=26 in inner function
        # But we can use broadcasting for speed
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist_mat = np.sqrt(dx**2 + dy**2)
        
        r_sum = r[:, None] + r[None, :]
        
        # We only care about i < j, but computing full matrix is easier to vectorize
        # dist_mat[i, j] is distance. We want dist_mat[i, j] >= r_sum[i, j]
        # Violation is max(0, r_sum - dist_mat)
        # Note: dist_mat[i,i] = 0, r_sum[i,i] = 2*r_i. This is not an overlap.
        # So we should mask the diagonal or ignore it.
        # Since we square it, and diagonal is 2*r_i, it will add huge penalty.
        # We must set diagonal of dist_mat to inf or handle it.
        np.fill_diagonal(dist_mat, np.inf)
        
        violations = np.maximum(0, r_sum - dist_mat)
        overlap_penalty = np.sum(violations**2)
        
        # 3. Penalty for boundary violations
        # x_i >= r_i, x_i <= 1 - r_i
        # y_i >= r_i, y_i <= 1 - r_i
        # Violation: max(0, r_i - x_i) + max(0, x_i - (1-r_i)) ...
        
        wall_pen = 0.0
        wall_pen += np.sum(np.maximum(0, r - x)**2)
        wall_pen += np.sum(np.maximum(0, x - (1 - r))**2)
        wall_pen += np.sum(np.maximum(0, r - y)**2)
        wall_pen += np.sum(np.maximum(0, y - (1 - r))**2)
        
        # Also ensure radii are positive
        # If r < 0, penalty
        wall_pen += np.sum(np.maximum(0, -r)**2)
        
        # Weighting
        # We want to maximize sum(r), but keep penalties low.
        # If penalties are 0, we are in a valid packing.
        # Start with moderate weights.
        weight_overlap = 100.0
        weight_wall = 100.0
        
        return val + weight_overlap * overlap_penalty + weight_wall * wall_pen

    # Initial guesses
    # We try multiple initializations to avoid local minima.
    # 1. Hexagonal-like grid
    # 2. Random
    # 3. Random with larger radii to encourage growth? No, start small.
    
    best_score = 0.0
    best_params = None
    
    def check_and_update(params):
        nonlocal best_score, best_params
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:]
        
        # Check validity
        # 1. Radii positive
        if np.any(r < 1e-6):
            return
        
        # 2. Inside square
        if np.any(x < r - 1e-5) or np.any(x > 1 - r + 1e-5):
            return
        if np.any(y < r - 1e-5) or np.any(y > 1 - r + 1e-5):
            return
        
        # 3. No overlaps
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist_mat = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, None] + r[None, :]
        np.fill_diagonal(dist_mat, np.inf)
        
        if np.any(dist_mat < r_sum - 1e-5):
            return
            
        score = np.sum(r)
        if score > best_score:
            best_score = score
            best_params = params.copy()

    def run_optimization(x0, y0, r0):
        params0 = np.concatenate([x0, y0, r0])
        
        # Bounds
        # x, y in [0, 1]
        # r in [0, 0.5] (max possible radius for one circle in unit square)
        bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
        
        try:
            res = minimize(
                objective, 
                params0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success or res.fun < 1e-3: # If objective is very low (close to 0 or negative)
                check_and_update(res.x)
        except Exception:
            pass

    # Initialization 1: Grid
    # Approximate hex packing
    # 26 circles. 
    # Rows: 5, 6, 5, 6, 4? 
    # Let's just use a simple grid 5x5 + 1 extra? 
    # Or 6x5 grid, drop one?
    # 6x5 = 30. Drop 4.
    # Let's use a dense random init with small radii.
    
    rng = np.random.default_rng(42)
    
    # Strategy: Multi-start with random positions and small initial radii.
    # The optimizer will grow radii as it finds space.
    # However, starting with small radii might lead to many small circles.
    # Better: Start with radii close to expected max (0.1) and let optimizer fix overlaps?
    # If we start with overlaps, the penalty is huge, and it might just shrink radii.
    # To maximize sum, we want it to grow.
    # L-BFGS-B minimizes. If we start with valid small circles, it will try to grow them
    # until they hit constraints.
    
    num_starts = 20
    
    for i in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        # Random positions
        x0 = rng.uniform(0.1, 0.9, N)
        y0 = rng.uniform(0.1, 0.9, N)
        
        # Initial radii: small, so they don't overlap initially
        # If we pick r too large, they overlap immediately.
        # r=0.05 is safe for 26 circles in 1x1? 
        # Area = 26 * pi * 0.05^2 ~ 0.2. Density 0.2. Safe.
        r0 = np.full(N, 0.05)
        
        # Add some noise to radii?
        # r0 = rng.uniform(0.04, 0.06, N)
        
        run_optimization(x0, y0, r0)
        
        # Alternative start: Grid based
        if i % 5 == 0:
            # Create a grid of points
            # 5x5 grid = 25 points. Add 1 more in center?
            # 5x5 grid spacing 0.2. Points at 0.1, 0.3, 0.5, 0.7, 0.9
            # Radius 0.1 would touch.
            # Let's try a grid init for some runs.
            n_grid = 5
            xs = np.linspace(0.1, 0.9, n_grid)
            ys = np.linspace(0.1, 0.9, n_grid)
            gx, gy = np.meshgrid(xs, ys)
            gx = gx.flatten()
            gy = gy.flatten()
            # We need 26. Take first 25, add one at center (0.5, 0.5) with small r?
            # Or just take 26 random points from a denser grid.
            # Let's stick to random for now, it's more robust to local minima if enough starts.
            pass

    # If we still have time, do a polish step on the best solution found
    if best_params is not None and time.time() < deadline - 5:
        # Use the best found as initial guess for a more aggressive run
        # Maybe increase penalty weights to force tighter packing?
        # Or just run again with tighter tolerances.
        
        # Let's try a second phase with higher penalty weights to squeeze out more radius
        # by forcing overlaps to zero more strictly.
        
        def objective_strict(params):
            x = params[:N]
            y = params[N:2*N]
            r = params[2*N:]
            
            val = -np.sum(r)
            
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist_mat = np.sqrt(dx**2 + dy**2)
            r_sum = r[:, None] + r[None, :]
            np.fill_diagonal(dist_mat, np.inf)
            
            violations = np.maximum(0, r_sum - dist_mat)
            overlap_penalty = np.sum(violations**2)
            
            wall_pen = 0.0
            wall_pen += np.sum(np.maximum(0, r - x)**2)
            wall_pen += np.sum(np.maximum(0, x - (1 - r))**2)
            wall_pen += np.sum(np.maximum(0, r - y)**2)
            wall_pen += np.sum(np.maximum(0, y - (1 - r))**2)
            wall_pen += np.sum(np.maximum(0, -r)**2)
            
            # Higher weights
            return val + 1000.0 * overlap_penalty + 1000.0 * wall_pen

        bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
        
        try:
            res = minimize(
                objective_strict, 
                best_params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            check_and_update(res.x)
        except Exception:
            pass

    if best_params is not None:
        x = best_params[:N]
        y = best_params[N:2*N]
        r = best_params[2*N:]
        return np.column_stack((x, y, r))
    else:
        # Fallback: Grid of small circles
        # 6x5 grid, take 26
        # Spacing 1/6 ~ 0.16. Radius 0.08
        n_x, n_y = 6, 5
        r_val = 0.07
        x_coords = np.linspace(r_val, 1-r_val, n_x)
        y_coords = np.linspace(r_val, 1-r_val, n_y)
        x_f, y_f = np.meshgrid(x_coords, y_coords)
        x_f = x_f.flatten()
        y_f = y_f.flatten()
        r_f = np.full(n_x*n_y, r_val)
        return np.column_stack((x_f[:N], y_f[:N], r_f[:N]))

# EVOLVE-BLOCK-END
