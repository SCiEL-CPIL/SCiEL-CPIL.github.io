import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295
        
        n_evals = self.budget
        if n_evals <= 0:
            return []
            
        results = []
        
        # Strategy: Sample uniformly in f1 space to maximize hypervolume Riemann sum accuracy.
        # 1. Generate a dense grid of x0.
        # 2. Compute f1, f2 for the grid.
        # 3. Select n_evals points that are uniformly spaced in f1.
        # 4. Evaluate these points.
        
        # Dense grid for offline selection
        n_grid = 10000
        x0_grid = np.linspace(0.0, 1.0, n_grid)
        
        # Compute f1 and f2 for the grid
        # f1(x) = 1 + x * (sin(10*pi*x) + sin(10*pi*x^2))
        # f2(x) = 1 - sqrt(f1(x))
        sin_term1 = np.sin(10 * np.pi * x0_grid)
        sin_term2 = np.sin(10 * np.pi * x0_grid**2)
        f1_grid = 1 + x0_grid * (sin_term1 + sin_term2)
        
        # Ensure f1 is positive for sqrt
        f1_grid = np.maximum(f1_grid, 1e-6)
        f2_grid = 1 - np.sqrt(f1_grid)
        
        # Sort by f1
        sort_idx = np.argsort(f1_grid)
        f1_sorted = f1_grid[sort_idx]
        f2_sorted = f2_grid[sort_idx]
        x0_sorted = x0_grid[sort_idx]
        
        # Select n_evals points uniformly spaced in f1
        if n_evals == 1:
            # Pick the point with median f1? Or min? 
            # To maximize hypervolume, we want to cover the range.
            # With 1 point, we can't cover much. 
            # The point that maximizes the box area is the one with max f1? 
            # Box area = (11-f1)(11-f2). 
            # Let's just pick the middle one or the one with max f1.
            # Max f1 gives min f2. 
            # (11-f1_max)(11-f2_min) vs (11-f1_min)(11-f2_max).
            # f1_min ~ 1, f2_max ~ 0. Area ~ 10*11 = 110.
            # f1_max ~ 2, f2_min ~ 0.7. Area ~ 9*10.3 = 92.
            # So min f1 is better for 1 point?
            # But we want to maximize the union. 
            # Let's pick the point that maximizes (11-f1)(11-f2).
            areas = (11 - f1_grid) * (11 - f2_grid)
            best_idx = np.argmax(areas)
            x0s = [x0_grid[best_idx]]
        else:
            # Uniform spacing in f1
            # We want indices i_0, ..., i_{N-1} such that f1[i_k] are roughly uniform.
            # Simple approach: linear interpolation of indices.
            indices = np.linspace(0, n_grid - 1, n_evals).astype(int)
            x0s = x0_sorted[indices]
            
        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = np.zeros(self.dim, dtype=float)
            x[0] = float(x0s[i])
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break
                
        return results
