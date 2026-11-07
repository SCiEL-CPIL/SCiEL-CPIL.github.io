"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-Start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin: 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Track best
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_made = 0
        
        # Helper to count calls and update best
        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return float('inf')
            # Clip to bounds
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val
        
        # Phase 1: Global Exploration via Grid
        # For 5D, a 3^5 = 243 point grid is feasible and provides good coverage.
        # This helps identify the basin of attraction of the global minimum.
        grid_points_per_dim = 3
        # Generate grid
        grid_axes = np.linspace(-5, 5, grid_points_per_dim)
        
        # Create all combinations
        # Using meshgrid for efficiency
        grids = np.meshgrid(*([grid_axes] * dim), indexing='ij')
        grid_points = np.array(grids).reshape(dim, -1).T
        
        # Evaluate grid points
        # We limit the number of grid points if budget is tight, but 243 is small.
        # Shuffle grid points to avoid bias towards corners if budget runs out?
        # Actually, systematic is better for coverage.
        # But we should prioritize center? No, uniform is safer for unknown shift.
        
        # To ensure we don't waste time if grid is too large, we can limit it.
        # But 243 calls is < 500.
        
        # We will evaluate all grid points if budget allows.
        # If budget is very small, we might need to subsample.
        # Given budget=500, we have ~250 calls left for local search.
        
        # Evaluate grid
        for x in grid_points:
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            evaluate(x)
            
        # Phase 2: Local Refinement via Nelder-Mead
        # Select top K grid points as starting points for local search.
        # We need to store the grid evaluation results to select top K.
        # Let's re-evaluate or store? Storing is better.
        # Let's redo grid evaluation with storage.
        
        # Clear previous state? No, we already evaluated. 
        # We didn't store the values per point in a list, only best.
        # Let's re-structure:
        # 1. Evaluate grid, store (f, x) pairs.
        # 2. Sort by f.
        # 3. Run Nelder-Mead on top K.
        
        # Since we already called evaluate which updated best, we lost the per-point data.
        # Let's restart the logic cleanly.
        
        # Reset for clean implementation
        calls_made = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        
        def evaluate2(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return float('inf')
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val
        
        # 1. Grid Search
        grid_points_per_dim = 3
        grid_axes = np.linspace(-5, 5, grid_points_per_dim)
        grids = np.meshgrid(*([grid_axes] * dim), indexing='ij')
        grid_points = np.array(grids).reshape(dim, -1).T
        
        grid_results = []
        for x in grid_points:
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            f_val = evaluate2(x)
            grid_results.append((f_val, x))
            
        # Sort by fitness
        grid_results.sort(key=lambda item: item[0])
        
        # 2. Local Search
        # Use top 5 starting points
        num_starts = 5
        # Budget for local search: remaining budget
        # Nelder-Mead is fast, but we need to be careful with maxiter.
        
        for i in range(num_starts):
            if i >= len(grid_results):
                break
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
                
            start_point = grid_results[i][1]
            
            # Run Nelder-Mead
            # We use a wrapper for func to count calls
            # SciPy's minimize doesn't easily allow call counting without wrapping
            # But our evaluate2 already counts.
            
            # Nelder-Mead options
            options = {
                'maxiter': 50, # Limit iterations to prevent hanging
                'xatol': 1e-6,
                'fatol': 1e-6,
                'adaptive': True
            }
            
            try:
                # Use a simple wrapper to ensure we clip and count
                def local_func(x):
                    if calls_made >= budget:
                        return float('inf')
                    if time.time() > deadline:
                        return float('inf')
                    return evaluate2(x)
                
                res = minimize(local_func, start_point, method='Nelder-Mead', options=options)
                
                # res.fun is the best value found by NM
                # res.x is the best point
                # evaluate2 already updated best_x and best_f
            except Exception:
                pass
                
        return best_x, best_f
# EVOLVE-BLOCK-END
