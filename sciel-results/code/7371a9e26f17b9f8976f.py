"""Optimizer for BBOB-style f14 (Different powers) on [-5, 5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start L-BFGS-B optimizer with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        bounds = [(-5.0, 5.0)] * self.dim

        # Track best solution found
        best_x = None
        best_f = np.inf

        # Use a subset of budget for L-BFGS-B restarts
        # Each L-BFGS-B call can use up to ~50 function evaluations
        # We'll do multiple restarts, each with a fresh random start

        # First, do a few quick evaluations to get a sense of the landscape
        # and potentially find a good starting point
        num_initial = min(10, self.budget // 10)
        for _ in range(num_initial):
            x = rng.uniform(-4, 4, self.dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Multi-start L-BFGS-B
        # Each restart uses a random starting point
        # We limit the number of function evaluations per restart
        max_iter_per_restart = 30
        num_restarts = min(20, (self.budget - num_initial) // max_iter_per_restart)

        for i in range(num_restarts):
            # Check deadline
            if time.time() > deadline:
                break

            # Choose starting point: mix of random and perturbation of best
            if i == 0 and best_x is not None:
                x0 = best_x.copy()
            elif i < 5 and best_x is not None:
                # Perturb the best solution
                x0 = best_x + rng.normal(0, 1.0, self.dim)
                x0 = np.clip(x0, -5, 5)
            else:
                x0 = rng.uniform(-4, 4, self.dim)

            # Define wrapper that counts function calls and respects budget
            call_count = [0]

            def wrapped_func(x):
                call_count[0] += 1
                if time.time() > deadline:
                    raise TimeoutError
                return func(np.clip(x, -5, 5))

            try:
                result = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': max_iter_per_restart, 'maxfun': max_iter_per_restart}
                )
                fx = result.fun
                if fx < best_f:
                    best_f = fx
                    best_x = np.clip(result.x, -5, 5).copy()
            except (TimeoutError, Exception):
                # If L-BFGS-B fails or times out, continue
                pass

        # If we still have budget, do a local refinement around the best point
        if best_x is not None and time.time() < deadline:
            # Try a few small perturbations around the best point
            for scale in [0.5, 0.2, 0.1, 0.05]:
                if time.time() > deadline:
                    break
                for _ in range(3):
                    if time.time() > deadline:
                        break
                    x = best_x + rng.normal(0, scale, self.dim)
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

        # Return the best solution found
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
