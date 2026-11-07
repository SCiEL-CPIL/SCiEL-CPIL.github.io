"""Coordinate-wise sequential search for Bent Cigar (f12) optimization."""
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """Two-phase optimization: Coarse Grid Search + L-BFGS-B."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 275.0
        n = self.dim
        budget = self.budget
        
        calls_used = 0
        best_x = None
        best_fx = np.inf

        def eval_func(x):
            nonlocal calls_used, best_x, best_fx
            if calls_used >= budget or time.time() > deadline:
                raise ValueError("Budget or time exceeded")
            calls_used += 1
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse Grid Search
        # Use a 3-level grid per dimension: -4, 0, 4
        # 3^5 = 243 points. Fits within budget 500.
        grid_points = np.array(np.meshgrid([-4.0, 0.0, 4.0] * n), dtype=float).T.reshape(-1, n)
        
        for p in grid_points:
            if calls_used >= budget or time.time() > deadline:
                break
            try:
                eval_func(p)
            except ValueError:
                break

        # If no points were evaluated (should not happen), fallback to random
        if best_x is None:
            best_x = rng.uniform(-5, 5, n)
            best_fx = eval_func(best_x)

        # Phase 2: L-BFGS-B Local Minimization
        # Start from the best point found in Phase 1
        x0 = best_x.copy()
        
        # Define bounds
        bounds = [(-5.0, 5.0)] * n
        
        # We need to stop if we exceed budget.
        # L-BFGS-B uses finite differences for gradients.
        # Each gradient evaluation costs ~2*N + 1 function calls.
        # We have roughly 250 calls left.
        
        try:
            res = scipy.optimize.minimize(
                eval_func, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100, 'maxls': 20}
            )
            # Update best if scipy found something better (it should, as it minimizes)
            if res.fun < best_fx:
                best_fx = res.fun
                best_x = res.x
        except ValueError:
            # Budget or time exceeded during optimization
            pass
        except Exception:
            # Catch other exceptions to ensure we return something
            pass

        return best_x, best_fx
