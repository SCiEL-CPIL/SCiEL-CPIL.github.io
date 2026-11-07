"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead with coarse grid for early progress."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # safety margin

        call_count = 0
        best_x = np.zeros(dim)
        best_fx = np.inf

        def record(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        def nm_func(x):
            nonlocal call_count
            if call_count >= budget:
                return best_fx
            val = func(x)
            call_count += 1
            record(x, val)
            return val

        rng = np.random.default_rng(42)

        # Phase 1: Coarse grid search for early progress
        n_per_dim = 3  # 3 points per dimension
        grid_vals = np.linspace(-4, 4, n_per_dim)
        
        # Generate grid points
        grid_points = []
        for idx in range(n_per_dim ** dim):
            point = np.array([grid_vals[(idx // (n_per_dim ** i)) % n_per_dim] for i in range(dim)])
            grid_points.append(point)
        
        # Shuffle grid points for random exploration
        rng.shuffle(grid_points)
        
        # Evaluate grid points until budget is exhausted or all done
        for point in grid_points:
            if call_count >= budget:
                break
            if time.time() > deadline - 10:
                break
            fx = func(point)
            call_count += 1
            record(point, fx)

        # Phase 2: Local optimization with Nelder-Mead from best point
        if call_count < budget and time.time() < deadline - 10:
            # Start NM from the best point found so far
            result = minimize(nm_func, best_x, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            record(result.x, result.fun)

            # If budget remains, try NM from a few other good starting points
            # Sort grid points by function value and try top few
            evaluated_points = []
            for point in grid_points:
                if call_count >= budget:
                    break
                if time.time() > deadline - 10:
                    break
                # Re-evaluate to get function value (we didn't store them separately)
                # Actually, let's just do a few more random restarts
                break
            
            # Try 2 more random restarts if budget allows
            for restart in range(2):
                if call_count >= budget:
                    break
                if time.time() > deadline - 10:
                    break
                
                # Start from a random point near the best
                start_x = best_x + rng.normal(0, 0.5, size=dim)
                start_x = np.clip(start_x, -5, 5)
                
                result = minimize(nm_func, start_x, method='Nelder-Mead',
                                  options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
                record(result.x, result.fun)

        return best_x, best_fx
# EVOLVE-BLOCK-END
