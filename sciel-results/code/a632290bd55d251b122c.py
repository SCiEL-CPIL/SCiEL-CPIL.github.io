"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio of (1+1)-ES restarts with simple 1/5-rule adaptation.

    Strategy:
      1. Four (1+1)-ES restarts with diverse random initializations and step sizes.
      2. Simple 1/5-rule step-size adaptation (1.5x on success, 0.87x on failure).
      3. Equal budget allocation across restarts for consistent diversity.
      4. Final local refinement around the best point found.
      5. Explicit wall-clock deadline guard.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0

        call_count = [0]

        def make_call(x):
            if call_count[0] >= budget:
                return None
            call_count[0] += 1
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            return float(func(x))

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        reserve = min(50, budget // 5)
        n_restarts = 4
        alloc_per_restart = (budget - reserve) // n_restarts

        init_points = [
            rng.uniform(-5, 5, dim),
            rng.uniform(-5, 5, dim),
            rng.uniform(-5, 5, dim),
            rng.uniform(-5, 5, dim),
        ]
        initial_sigmas = [2.0, 1.5, 3.0, 1.0]

        for r in range(n_restarts):
            if call_count[0] >= budget - reserve:
                break
            if time.time() > deadline - 20:
                break

            x = init_points[r]
            fx = make_call(x)
            if fx is None:
                break
            update_best(x, fx)

            sigma = initial_sigmas[r]

            for _ in range(alloc_per_restart):
                if call_count[0] >= budget - reserve:
                    break
                if time.time() > deadline - 20:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = make_call(y)
                if fy is None:
                    break

                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.5
                else:
                    sigma *= 0.87

                sigma = float(np.clip(sigma, 0.01, 10.0))

        if best_x is not None and call_count[0] < budget - 5 and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            remaining_calls = budget - call_count[0] - 5

            for _ in range(remaining_calls):
                if call_count[0] >= budget - 5:
                    break
                if time.time() > deadline - 5:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = make_call(y)
                if fy is None:
                    break

                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.2
                else:
                    sigma *= 0.8

                sigma = float(np.clip(sigma, 0.001, 1.0))

        return best_x, best_f
# EVOLVE-BLOCK-END
