"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead optimizer for global and local search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        calls_made = 0
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        evals = []  # List of (f, x)

        def tracked_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return 0.0
            val = func(x)
            calls_made += 1
            evals.append((val, x.copy()))
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Global Exploration via Random Sampling
        # Use 20% of budget for sampling to find good basins
        n_samples = max(10, self.budget // 5)
        samples = rng.uniform(-5, 5, size=(n_samples, self.dim))
        for i in range(n_samples):
            tracked_func(samples[i])

        # Sort evaluated points by fitness
        evals.sort(key=lambda item: item[0])
        
        # Phase 2: Local Refinement using Nelder-Mead
        # Select top 3 distinct starting points
        start_points = []
        last_point = None
        for f, x in evals:
            if last_point is None or not np.allclose(x, last_point, atol=1e-5):
                start_points.append(x)
                last_point = x
            if len(start_points) >= 3:
                break

        if not start_points:
            return best_x, best_f

        # Allocate remaining budget to Nelder-Mead runs
        remaining_budget = self.budget - calls_made
        # Allocate 50% to best, 30% to second, 20% to third
        allocs = [0.5, 0.3, 0.2]
        
        for i, start in enumerate(start_points):
            if calls_made >= self.budget - 5:
                break
            
            # Estimate max iterations for Nelder-Mead
            # Nelder-Mead uses roughly dim+1 evals per iteration
            # Let's just let it run until budget is nearly exhausted or it converges
            # We use maxiter to limit it, but tracked_func enforces budget.
            
            try:
                result = minimize(
                    tracked_func,
                    start,
                    method='Nelder-Mead',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 200, 
                        'xatol': 1e-8, 
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
