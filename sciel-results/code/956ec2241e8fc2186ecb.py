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

        # Phase 1: Single (1+1)-ES with 1/5-rule for most of the budget
        # This follows the simpler, higher-scoring approach from inspiration 2
        x = rng.uniform(-5, 5, dim)
        fx = make_call(x)
        if fx is None:
            return None, np.inf
        update_best(x, fx)

        sigma = 2.0
        single_es_budget = budget - 80  # Reserve 80 for restarts + polish

        for _ in range(single_es_budget):
            if call_count[0] >= budget - 80:
                break
            if time.time() > deadline - 30:
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

        # Phase 2: Two short restarts from different points
        restart_budget = 30
        restart_inits = [
            rng.uniform(-5, 5, dim),
            rng.uniform(-5, 5, dim),
        ]
        restart_sigmas = [3.0, 1.5]

        for r in range(2):
            if call_count[0] >= budget - 50:
                break
            if time.time() > deadline - 20:
                break

            x = restart_inits[r]
            fx = make_call(x)
            if fx is None:
                break
            update_best(x, fx)

            sigma = restart_sigmas[r]

            for _ in range(restart_budget):
                if call_count[0] >= budget - 50:
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

        # Phase 3: Final polish around the best point found
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
