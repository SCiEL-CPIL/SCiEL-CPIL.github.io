"""Multi-Start Nelder-Mead Optimizer."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-Start Nelder-Mead optimizer for the BBOB-style f16 (Weierstrass) task."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        calls = [0]
        best_f = float('inf')
        best_x = np.zeros(dim)

        class BudgetExceeded(Exception):
            pass

        def counting_func(x):
            nonlocal best_f, best_x
            calls[0] += 1
            if calls[0] > budget:
                raise BudgetExceeded()
            x = np.asarray(x)
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Generate initial points
        n_starts = 10
        rng = np.random.default_rng(42)
        
        initial_points = []
        # Center
        initial_points.append(np.zeros(dim))
        # Random points
        for _ in range(n_starts - 1):
            initial_points.append(rng.uniform(-5, 5, dim))
            
        # Evaluate initial points
        initial_fits = []
        for p in initial_points:
            try:
                f = counting_func(p)
                initial_fits.append(f)
            except BudgetExceeded:
                return best_x, best_f
            except Exception:
                initial_fits.append(float('inf'))
                
        if calls[0] >= budget:
            return best_x, best_f

        # Sort initial points by fitness
        order = np.argsort(initial_fits)
        
        for idx in order:
            if calls[0] >= budget:
                break
            
            start_point = initial_points[idx]
            
            # Run Nelder-Mead
            try:
                result = minimize(
                    counting_func,
                    start_point,
                    method='Nelder-Mead',
                    options={'maxiter': budget - calls[0], 'xatol': 1e-6, 'fatol': 1e-6}
                )
            except BudgetExceeded:
                pass
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
