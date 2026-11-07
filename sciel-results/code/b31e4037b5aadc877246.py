import numpy as np
import scipy.optimize as opt
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        best_x = None
        best_f = np.inf
        used_calls = 0
        max_calls = self.budget
        
        def safe_func(x):
            nonlocal used_calls, best_f, best_x
            if used_calls >= max_calls:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("Time limit exceeded")
            used_calls += 1
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val
        
        # Phase 1: Coarse Grid Search (3^5 = 243 points)
        # Katsuura has fractal structure; a coarse grid helps locate the basin
        grid_size = 3
        grid_points = np.array(np.meshgrid(*[np.linspace(-5, 5, grid_size) for _ in range(self.dim)])).T.reshape(-1, self.dim)
        
        for p in grid_points:
            if used_calls >= max_calls or time.time() > deadline:
                break
            try:
                safe_func(p)
            except RuntimeError:
                break
        
        if best_x is None:
            return None, np.inf

        # Phase 2: Recursive Refinement
        # Refine the best region with progressively finer grids
        # Start from the best point found in Phase 1
        center = best_x.copy()
        
        # Define refinement stages: grid size and radius
        # Stage 1: 5^5 grid in a small box around best point
        # Stage 2: 5^5 grid in an even smaller box
        # We adaptively choose the box size based on the function's scale
        
        # Katsuura's oscillations have a characteristic scale.
        # Let's try a few refinement levels.
        
        refinement_configs = [
            (5, 1.0),   # 5^5=3125 points, radius 1.0
            (5, 0.3),   # 5^5=3125 points, radius 0.3
            (7, 0.1),   # 7^5=16807 points, radius 0.1
            (9, 0.05),  # 9^5=59049 points, radius 0.05
        ]
        
        for grid_sz, radius in refinement_configs:
            if used_calls >= max_calls or time.time() > deadline:
                break
            
            # Generate grid points in a box around the current best
            # The box is [center - radius, center + radius]
            # But we need to clip to [-5, 5]
            lower = np.clip(center - radius, -5, 5)
            upper = np.clip(center + radius, -5, 5)
            
            # If the box is degenerate (lower == upper), skip
            if np.allclose(lower, upper):
                break
            
            # Generate grid
            grid_points = np.array(np.meshgrid(*[np.linspace(lower[i], upper[i], grid_sz) for i in range(self.dim)])).T.reshape(-1, self.dim)
            
            # Evaluate points
            for p in grid_points:
                if used_calls >= max_calls or time.time() > deadline:
                    break
                try:
                    safe_func(p)
                except RuntimeError:
                    break
            
            # If budget is exhausted, break
            if used_calls >= max_calls:
                break

        # Phase 3: Local Polish with Nelder-Mead
        if used_calls < max_calls and time.time() < deadline:
            final_budget = max_calls - used_calls
            if final_budget > 10:
                try:
                    result = opt.minimize(
                        safe_func, 
                        best_x, 
                        method='Nelder-Mead',
                        options={
                            'maxiter': final_budget,
                            'maxfev': final_budget,
                            'xatol': 1e-10,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                    # best_f and best_x are updated in safe_func
                except (RuntimeError, Exception):
                    pass

        return best_x, best_f
