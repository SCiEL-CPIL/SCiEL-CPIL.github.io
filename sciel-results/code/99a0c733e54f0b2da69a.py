"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

class BudgetExceeded(Exception):
    pass

class Optimizer:
    """Robust multi-start local optimizer for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        call_count = 0
        max_calls = self.budget
        dim = self.dim

        # Wrapper to count calls and enforce budget
        def counted_func(x):
            nonlocal call_count
            if call_count >= max_calls:
                raise BudgetExceeded()
            call_count += 1
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)

        best_x = np.zeros(dim)
        best_f = float('inf')

        # Initial evaluation
        try:
            best_f = counted_func(best_x)
        except BudgetExceeded:
            return best_x, best_f

        # Strategy: Multi-start Nelder-Mead
        # Budget is 500. NM in 5D takes ~50-100 calls to converge.
        # We can do ~5-8 restarts.
        num_restarts = 5

        for i in range(num_restarts):
            if call_count >= max_calls or time.time() > deadline:
                break

            # Random start point in the box
            x_start = rng.uniform(-5, 5, size=dim)

            # Check time/budget before starting
            if call_count >= max_calls or time.time() > deadline:
                break

            try:
                # Use Nelder-Mead
                result = scipy.optimize.minimize(
                    counted_func,
                    x_start,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4}
                )

                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()

            except BudgetExceeded:
                # Budget exhausted, stop
                break
            except Exception:
                # If optimization fails, continue to next restart
                pass

        # If we have budget left, do a final local refinement from the best point
        if call_count < max_calls and time.time() < deadline:
            try:
                result = scipy.optimize.minimize(
                    counted_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-5, 'fatol': 1e-5}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except BudgetExceeded:
                pass
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
