"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, deadline guard, and multi-restart."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_fx = np.inf
        calls = 0

        # Multi-restart (1+1)-ES
        # Allocate budget across restarts: first run gets more, later ones less
        # For Schwefel, multiple restarts help escape local minima
        n_restarts = 5
        # Budget allocation: geometric decay
        weights = [0.4, 0.25, 0.15, 0.1, 0.1]
        for ri in range(n_restarts):
            if calls >= budget:
                break
            if time.time() >= deadline - 2:
                break

            # Initial point: random for restarts > 0, near best for restart 0
            if ri == 0:
                x = rng.uniform(-5, 5, dim)
            else:
                # Start near the current best with some perturbation
                if best_x is not None:
                    x = best_x + rng.uniform(-1.5, 1.5, dim)
                    x = np.clip(x, -5, 5)
                else:
                    x = rng.uniform(-5, 5, dim)

            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            sigma = 2.0
            # Run (1+1)-ES with remaining budget for this restart
            alloc = int(budget * weights[ri])
            for _ in range(alloc):
                if calls >= budget:
                    break
                if time.time() >= deadline - 2:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

                # Clamp sigma to avoid numerical issues
                if sigma < 1e-8:
                    sigma = 1e-8
                if sigma > 10.0:
                    sigma = 10.0

        return best_x, best_fx
# EVOLVE-BLOCK-END
