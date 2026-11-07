"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Powell optimizer for black-box continuous functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        calls_made = [0]
        
        def wrapped_func(x):
            if calls_made[0] >= self.budget:
                return 1e30
            calls_made[0] += 1
            return func(x)

        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf

        # Dynamic restart loop to utilize full budget
        # Powell is derivative-free and robust to ill-conditioning
        max_calls_per_restart = 50
        
        while calls_made[0] < self.budget - max_calls_per_restart:
            x0 = rng.uniform(-5, 5, self.dim)
            
            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='Powell',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 100,
                        'maxfev': max_calls_per_restart,
                        'ftol': 1e-12,
                        'xtol': 1e-12
                    }
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                continue

        # Use any remaining budget for one final attempt
        if calls_made[0] < self.budget:
            x0 = rng.uniform(-5, 5, self.dim)
            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='Powell',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 100,
                        'maxfev': self.budget - calls_made[0],
                        'ftol': 1e-12,
                        'xtol': 1e-12
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
