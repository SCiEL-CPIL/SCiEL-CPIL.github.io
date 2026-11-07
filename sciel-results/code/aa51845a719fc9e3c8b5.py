"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools


class Optimizer:
    """Deterministic grid search + fixed-step local hill climb."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill
        n = self.dim
        budget = self.budget
        calls = 0

        def safe_call(x):
            nonlocal calls
            if calls >= budget:
                return float('inf')
            calls += 1
            return func(x)

        # Phase 1: Coarse grid search
        # Minimizer is in [-4, 4]^5. Use levels -4, 0, 4.
        levels = [-4.0, 0.0, 4.0]
        grid_points = list(itertools.product(levels, repeat=n))
        
        # We store (f, x) for sorting
        results = []
        for pt in grid_points:
            if calls >= budget or time.time() > deadline:
                break
            x = np.array(pt)
            f = safe_call(x)
            results.append((f, x))
            
        if not results:
            # Fallback
            x0 = np.zeros(n)
            f0 = safe_call(x0)
            best_x = x0
            best_f = f0
            return best_x, best_f

        # Sort by function value
        results.sort(key=lambda item: item[0])
        
        best_x = results[0][1]
        best_f = results[0][0]
        
        # Phase 2: Fixed-step local hill climb from the best grid point
        # Use a series of decreasing step sizes to refine the solution
        step_sizes = [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
        
        for step_size in step_sizes:
            if calls >= budget or time.time() > deadline:
                break
            
            # Try moving in each coordinate direction (positive and negative)
            improved = True
            while improved and calls < budget and time.time() <= deadline:
                improved = False
                for axis in range(n):
                    for direction in [1, -1]:
                        if calls >= budget or time.time() > deadline:
                            break
                        
                        x_new = best_x.copy()
                        x_new[axis] += direction * step_size
                        
                        # Clip to the box [-5, 5]
                        x_new = np.clip(x_new, -5.0, 5.0)
                        
                        f_new = safe_call(x_new)
                        
                        if f_new < best_f:
                            best_f = f_new
                            best_x = x_new
                            improved = True
        
        return best_x, best_f
# EVOLVE-BLOCK-END
