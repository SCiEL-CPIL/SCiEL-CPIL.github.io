"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start L-BFGS-B with time guard for Discus (f11)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        call_count = 0

        def wrapped(x):
            nonlocal call_count
            call_count += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            return func(x_clipped)

        # Generate initial points: center + grid-like spread + random
        rng = np.random.default_rng(42)
        initial_points = []
        
        # Center
        initial_points.append(np.zeros(dim))
        
        # Grid points at -3, 0, 3 in each dimension (but too many for dim=5: 3^5=243)
        # Use a smaller set: random uniform points
        n_random = 8
        for _ in range(n_random):
            initial_points.append(rng.uniform(-4, 4, dim))
        
        # Add some structured points
        for i in range(dim):
            for val in [-3.0, 3.0]:
                pt = np.zeros(dim)
                pt[i] = val
                initial_points.append(pt)
        
        initial_points = np.array(initial_points)
        
        best_x = initial_points[0].copy()
        best_f = wrapped(best_x)
        
        bounds = [(-5.0, 5.0)] * dim
        
        for x0 in initial_points:
            if time.time() > deadline - 10.0:
                break
            if call_count >= budget:
                break
            
            try:
                result = minimize(
                    wrapped,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'maxfun': 100, 'gtol': 1e-8, 'ftol': 1e-12}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
        
        # If we have budget left, do a final local polish from the best point
        if time.time() < deadline - 5.0 and call_count < budget:
            try:
                result = minimize(
                    wrapped,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'maxfun': budget - call_count, 'gtol': 1e-10, 'ftol': 1e-14}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
