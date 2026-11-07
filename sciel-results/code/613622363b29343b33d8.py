"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with 1/5 success rule and a wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start = time.perf_counter()
        deadline = start + 285.0          # safety margin well under the 330 s kill
        rng = np.random.default_rng(0)
        budget = self.budget
        dim = self.dim

        # Split the budget into independent restarts; more restarts -> better
        # chance of escaping local minima and finding a good optimum early.
        n_restarts = 8
        per = max(1, budget // n_restarts)

        global_best_x = None
        global_best_f = np.inf

        for r in range(n_restarts):
            # Random initial point inside the box
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls = 1
            sigma = 2.0
            it = 0
            while calls < per:
                if time.perf_counter() > deadline:
                    break
                it += 1
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.5
                else:
                    sigma *= 0.87
                if fx < global_best_f:
                    global_best_f = fx
                    global_best_x = x.copy()
            if time.perf_counter() > deadline:
                break

        if global_best_x is None:
            x0 = np.zeros(dim)
            global_best_x = x0
            global_best_f = func(x0)

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
