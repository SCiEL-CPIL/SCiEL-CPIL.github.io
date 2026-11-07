import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        eval_count = 0
        
        def counting_func(x):
            nonlocal eval_count, best_f, best_x
            if eval_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
            val = float(func(x_clipped))
            eval_count += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        # Phase 1: Coarse Grid Search (3^5)
        coords_coarse = [-5.0, 0.0, 5.0]
        grid_points = np.array(np.meshgrid(*([coords_coarse]*dim))).T.reshape(-1, dim)
        
        for pt in grid_points:
            if eval_count >= budget or time.time() > deadline:
                break
            counting_func(pt)
            
        # Phase 2: Refine the best region with a finer Grid (3^5)
        # The best region is the cell containing the best point.
        # We define the refinement box around the best point.
        # Coarse step is 5.0. We use a box of size 5.0 (step 5.0/3? No, we just center a new grid).
        # Let's center a new 3^5 grid on the best_x found so far.
        # The extent of the new grid should be comparable to the coarse cell size.
        # Coarse cell size is 5.0 (from -5 to 0, or 0 to 5).
        # So we create a grid in [best_x - 2.5, best_x + 2.5] in each dimension?
        # Or [best_x - 5, best_x + 5]?
        # If we use [best_x - 2.5, best_x + 2.5], the step is 5.0/3? No.
        # If we use 3 points in a range of size 5.0, the points are at -2.5, 0, 2.5 relative to center.
        # This covers the immediate neighborhood.
        
        if best_x is not None and eval_count < budget:
            # Define the refinement box
            # Use a step size of 1.0 for the refinement grid?
            # 3 points in a box of size 2.0? Step 1.0.
            # Points: center - 1.0, center, center + 1.0.
            # This is a zoom-in.
            
            half_range = 2.5 # Matches coarse cell half-width
            coords_fine = [[best_x[i] - half_range, best_x[i], best_x[i] + half_range] for i in range(dim)]
            # Ensure bounds
            for i in range(dim):
                coords_fine[i] = np.clip(coords_fine[i], -5.0, 5.0)
            
            grid_points_fine = np.array(np.meshgrid(*coords_fine)).T.reshape(-1, dim)
            
            for pt in grid_points_fine:
                if eval_count >= budget or time.time() > deadline:
                    break
                counting_func(pt)

        # Phase 3: Polish with Nelder-Mead
        remaining = budget - eval_count
        if remaining > 10 and best_x is not None:
            try:
                res_nm = scipy.optimize.minimize(
                    counting_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining // 2,
                        'xatol': 1e-12,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
            except Exception:
                pass
        
        # Ensure we have a valid result
        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = float(func(best_x))
                eval_count += 1
            except Exception:
                best_f = 0.0
        
        return best_x, best_f
