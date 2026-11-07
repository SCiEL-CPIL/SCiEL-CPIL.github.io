"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with restarts for bimodal BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 330s kill

        # Phase 1: Initial random sampling to find a good starting point
        n_initial = min(10, budget // 5)
        best_x = None
        best_f = np.inf
        for _ in range(n_initial):
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 2: (1+1)-ES with restarts
        calls_used = n_initial
        max_no_improve = 40  # restart if no improvement for this many steps
        min_sigma = 0.01
        max_sigma = 5.0

        while calls_used < budget:
            # Check time
            if time.time() > deadline:
                break

            x = best_x.copy()
            fx = best_f
            sigma = 1.0
            no_improve = 0
            local_best_x = x.copy()
            local_best_f = fx

            # Run (1+1)-ES until no_improve threshold or budget exhausted
            while calls_used < budget and no_improve < max_no_improve:
                if time.time() > deadline:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.2, max_sigma)
                    no_improve = 0
                    if fx < local_best_f:
                        local_best_f = fx
                        local_best_x = x.copy()
                else:
                    sigma = max(sigma * 0.8, min_sigma)
                    no_improve += 1

            # Update global best
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

            # If we used all budget, break
            if calls_used >= budget:
                break

            # Otherwise, restart from a new random point (but keep best)
            # The restart is implicit: next iteration starts from best_x

        return best_x, best_f
# EVOLVE-BLOCK-END
