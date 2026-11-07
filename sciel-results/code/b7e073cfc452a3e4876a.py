"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """General-purpose optimizer using Nelder-Mead for smooth functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0

        calls = 0
        x_best = np.zeros(dim)
        fx_best = func(x_best)
        calls += 1

        def time_ok():
            return time.time() < deadline

        def wrapped_func(x):
            nonlocal calls
            if not time_ok() or calls >= budget:
                return fx_best  # Return current best to stop progress? Or large value?
                # Returning fx_best might cause stagnation.
                # Better to just let it run and check time later?
                # If we exceed budget, the result is invalid.
                # So we must stop.
                # We can't easily stop minimize.
                # We rely on maxiter.
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls += 1
            if fx < fx_best:
                fx_best = fx
                x_best = x_clipped.copy()
            return fx

        # Initial simplex
        # Start at origin, with a small step
        x0 = np.zeros(dim)
        # Nelder-Mead options
        # maxiter is number of iterations, not calls.
        # Each iteration uses ~1-3 calls.
        # Budget 500. Let's allow 200 iterations.
        res = minimize(
            wrapped_func,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 200,
                'xatol': 1e-6,
                'fatol': 1e-6,
                'adaptive': True
            }
        )

        # If the optimization finished early or didn't use all budget,
        # we might do a local polish or just return best.
        # The wrapped_func updates x_best and fx_best.
        
        return x_best, fx_best
# EVOLVE-BLOCK-END
