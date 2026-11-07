"""Multi-start Powell's method optimizer for BBOB-style Bent Cigar (f12)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Powell's method for ill-conditioned functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 290.0

        calls_made = 0
        best_x = None
        best_fx = np.inf

        # Counting wrapper
        def counting_func(x):
            nonlocal calls_made
            calls_made += 1
            return func(x)

        # Generate initial points: origin + random points spread across domain
        n_starts = 8
        starts = [np.zeros(dim)]
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-5, 5, dim))

        # For each start, run Powell's method with a budget allocation
        budget_per_start = max(10, budget // n_starts)

        for start in starts:
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break

            # Check if this start is better than current best
            f_start = counting_func(start)
            if f_start < best_fx:
                best_fx = f_start
                best_x = start.copy()

            # If calls are too low, skip optimization from this start
            if calls_made >= budget:
                break

            # Run Powell's method from this start
            try:
                result = minimize(
                    counting_func,
                    start,
                    method='Powell',
                    options={'maxiter': 100, 'maxfev': budget - calls_made}
                )
                if result.fun < best_fx:
                    best_fx = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

            # If we've used all budget, break
            if calls_made >= budget:
                break

        # If best_x is still None (shouldn't happen), use origin
        if best_x is None:
            best_x = np.zeros(dim)
            best_fx = func(best_x)
            calls_made += 1

        return best_x, best_fx
# EVOLVE-BLOCK-END
