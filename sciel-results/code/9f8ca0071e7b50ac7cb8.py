"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES portfolio with progressive refinement for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget

        def time_left():
            return time.time() < deadline

        call_count = [0]

        def safe_func(x):
            call_count[0] += 1
            return func(np.clip(x, -5.0, 5.0))

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        # Phase 1: Multi-restart portfolio
        n_restarts = 10
        refine_budget = 50
        total_search_budget = budget - 1 - refine_budget
        calls_per_restart = max(1, total_search_budget // n_restarts)

        for restart_idx in range(n_restarts):
            if not time_left() or call_count[0] >= budget - 5:
                break

            # Initialize: first restart random, subsequent around best
            if best_x is None:
                x = rng.uniform(-5, 5, dim)
            else:
                x = best_x + rng.normal(0, 1.5, dim)
                x = np.clip(x, -5, 5)

            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Adaptive sigma: start larger for exploration, smaller for later restarts
            sigma = 2.5 if restart_idx < 3 else 1.5

            for step in range(calls_per_restart):
                if not time_left() or call_count[0] >= budget - 5:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = safe_func(y)

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.35
                else:
                    sigma *= 0.82

                sigma = np.clip(sigma, 0.01, 5.0)

        # Phase 2: Coarse local refinement around global best
        if best_x is not None and time_left() and call_count[0] < budget - 2:
            x = best_x.copy()
            fx = best_f
            sigma = 0.8
            remaining = min(refine_budget, budget - 2 - call_count[0])

            for step in range(remaining):
                if not time_left() or call_count[0] >= budget - 2:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = safe_func(y)

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.8

                sigma = np.clip(sigma, 0.001, 1.5)

        # Phase 3: Fine-grained refinement
        if best_x is not None and time_left() and call_count[0] < budget - 1:
            x = best_x.copy()
            fx = best_f
            sigma = 0.15
            remaining = budget - 1 - call_count[0]

            for step in range(remaining):
                if not time_left() or call_count[0] >= budget - 1:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = safe_func(y)

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.15
                else:
                    sigma *= 0.80

                sigma = np.clip(sigma, 0.0001, 0.5)

        return best_x, best_f
# EVOLVE-BLOCK-END
