"""Optimizer for BBOB-style functions using hybrid multi-start L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid multi-start L-BFGS-B optimizer for 5D BBOB functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # Strategy: Use a mix of random starts and L-BFGS-B local optimization
        # For 5D with 300 function calls, we can afford ~10-20 L-BFGS-B runs
        # each using ~15-30 function calls

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        # Number of random starts to try
        # Each L-BFGS-B call typically uses ~20-30 evaluations in 5D
        # With 300 budget, we can do about 10-15 local optimizations
        n_starts = 12

        # Reserve some budget for initial random sampling
        # First, do a few random evaluations to get a sense of the landscape
        n_random = min(5, budget // 10)

        for i in range(n_random):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Now do L-BFGS-B from the best points and some random points
        # Use the best random point and a few others as starting points
        start_points = [best_x.copy()] if best_x is not None else []

        # Add a few more random starting points
        for i in range(min(4, n_starts - 1)):
            x = rng.uniform(-5, 5, dim)
            start_points.append(x)

        # Run L-BFGS-B from each starting point
        for start_idx, x0 in enumerate(start_points):
            if time.time() > deadline:
                break

            # Limit the number of function evaluations per L-BFGS-B run
            # to ensure we don't blow the budget
            maxiter = 50
            maxfev = min(25, max(10, budget // n_starts))

            try:
                result = minimize(
                    func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': maxiter, 'maxfev': maxfev, 'gtol': 1e-6}
                )
            except Exception:
                continue

            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x.copy()

            # If we have budget left, try another random start
            if start_idx < n_starts - 1 and time.time() < deadline:
                # Check if we're making progress
                x = rng.uniform(-5, 5, dim)
                start_points.append(x)

        # If we still have budget, do some final random search around the best point
        if best_x is not None and time.time() < deadline:
            # Local random perturbation around best point
            n_local = min(20, max(5, (budget - 100) // 5))
            sigma = 0.5
            for i in range(n_local):
                if time.time() > deadline:
                    break
                x = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.9

        # Return the best result found
        if best_x is None:
            # Fallback: return a random point
            best_x = rng.uniform(-5, 5, dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
