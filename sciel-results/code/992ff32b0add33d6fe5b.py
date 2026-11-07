import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start CMA-ES with adaptive restarts and L-BFGS-B polish for BBOB functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Safety margin: 330s limit, stop at 325s
        deadline = time.time() + 325

        best_x = None
        best_f = np.inf
        calls = 0

        bounds = [(-5.0, 5.0) for _ in range(dim)]
        lower = np.full(dim, -5.0)
        upper = np.full(dim, 5.0)

        # Wrapper to track calls and update best
        def wrapped_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            if x.shape != (dim,):
                x = x.reshape(-1)
            if len(x) != dim:
                x = x[:dim]
            val = func(x)
            if not np.isfinite(val):
                return 1e10
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        def polish(x_start, maxfev=20):
            """Run L-BFGS-B polish from a starting point."""
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return
            try:
                res = minimize(
                    wrapped_func,
                    x_start.copy(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 50,
                        'maxfev': maxfev,
                        'ftol': 1e-15,
                        'gtol': 1e-12,
                        'disp': False
                    }
                )
            except Exception:
                pass

        # Phase 1: Coarse Global Grid Search
        # Use a 2x2x2x2x2 grid = 32 points to start, covering the whole space.
        grid_size_1 = 2
        coords_1 = np.linspace(-5, 5, grid_size_1)
        axes_1 = np.meshgrid(*([coords_1] * dim), indexing='ij')
        grid_points_1 = np.stack([a.flatten() for a in axes_1], axis=1)
        
        results_1 = []
        for x in grid_points_1:
            if calls >= budget or time.time() > deadline:
                break
            val = wrapped_func(x)
            results_1.append((val, x))
            
        if not results_1:
            return np.zeros(dim), np.inf
            
        # Sort to find best
        results_1.sort(key=lambda t: t[0])
        best_val_1, best_x_1 = results_1[0]
        
        # Phase 2: Adaptive Local Refinement
        # Define a local box around the best point found in Phase 1
        radius = 2.0 
        local_lower = np.clip(best_x_1 - radius, -5, 5)
        local_upper = np.clip(best_x_1 + radius, -5, 5)
        
        # Ensure box has some volume
        if np.any(local_upper <= local_lower):
             local_lower = np.clip(best_x_1 - 0.5, -5, 5)
             local_upper = np.clip(best_x_1 + 0.5, -5, 5)
             
        # Use a 3x3x3x3x3 grid = 243 points for refinement
        grid_size_2 = 3
        max_calls_allowed = budget - calls
        if grid_size_2 ** dim > max_calls_allowed:
             grid_size_2 = 2 # Fallback to 32 points if budget is tight
             
        coords_2 = np.linspace(local_lower, local_upper, grid_size_2)
        axes_2 = np.meshgrid(*([coords_2] * dim), indexing='ij')
        grid_points_2 = np.stack([a.flatten() for a in axes_2], axis=1)
        
        for x in grid_points_2:
            if calls >= budget or time.time() > deadline:
                break
            wrapped_func(x)
            
        # Phase 3: Final Polish
        if calls < budget and best_x is not None and time.time() < deadline:
            remaining = budget - calls
            if remaining > 10:
                polish(best_x, maxfev=remaining)
                
        return best_x, best_f
