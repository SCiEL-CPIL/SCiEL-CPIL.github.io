"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """L-BFGS-B based optimizer with multiple starting points for Discus (f11)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Count function evaluations
        call_count = [0]

        def counted_func(x):
            if time.time() > deadline:
                raise TimeoutError
            val = func(np.asarray(x, dtype=np.float64))
            call_count[0] += 1
            return float(val)

        # Generate a grid of starting points
        # Use a combination of center and spread points
        n_starts = 10
        rng = np.random.default_rng(123)

        # Create starting points: mix of deterministic and random
        starts = []
        # Center point
        starts.append(np.zeros(dim))
        # Small offsets
        for i in range(1, n_starts):
            scale = 0.5 * (1 + i / n_starts)
            pt = rng.uniform(-scale, scale, dim)
            starts.append(pt)

        best_x = None
        best_f = np.inf

        # Budget allocation: give each start roughly budget // n_starts evaluations
        # But we need to be careful about the total budget
        evals_per_start = max(1, (budget - 1) // n_starts)

        for i, x0 in enumerate(starts):
            if time.time() > deadline:
                break
            if call_count[0] >= budget:
                break

            # Clip starting point to box
            x0 = np.clip(x0, -5, 5)

            # Run L-BFGS-B
            try:
                res = minimize(
                    counted_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={
                        'maxiter': evals_per_start,
                        'maxfun': evals_per_start,
                        'gtol': 1e-8,
                        'ftol': 1e-12,
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except (TimeoutError, Exception):
                # If timeout or error, try to use the best found so far
                break

        # If we still have budget remaining, do a final polish from the best point
        if best_x is not None and call_count[0] < budget and time.time() < deadline:
            remaining = budget - call_count[0]
            try:
                res = minimize(
                    counted_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={
                        'maxiter': remaining,
                        'maxfun': remaining,
                        'gtol': 1e-10,
                        'ftol': 1e-14,
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except (TimeoutError, Exception):
                pass

        # Ensure we return something valid
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
