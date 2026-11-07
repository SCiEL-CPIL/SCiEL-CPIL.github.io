"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive sigma and time guard for BBOB f18."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill
        n = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Number of restarts: use a few restarts to escape local minima
        # For dim=5, use 5 restarts with ~100 evals each
        n_restarts = 5
        evals_per_restart = max(10, budget // n_restarts)

        rng = np.random.default_rng(42)

        for r in range(n_restarts):
            if calls >= budget:
                break
            if time.time() > deadline - 5.0:
                break

            # Initialize: first restart at origin, others random
            if r == 0:
                x = np.zeros(n)
            else:
                x = rng.uniform(-5, 5, n)

            fx = evaluate(x)
            sigma = 1.5  # initial step size
            stagnation = 0
            max_stagnation = 30

            remaining = budget - calls
            for _ in range(evals_per_restart):
                if calls >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break

                # Generate offspring
                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)

                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.3  # increase step on success
                    stagnation = 0
                else:
                    sigma *= 0.85  # decrease step on failure
                    stagnation += 1

                # If too much stagnation, break to next restart
                if stagnation >= max_stagnation:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
