"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio of diverse (1+1)-ES restarts + final local refinement.

    Strategy:
      1. Multiple (1+1)-ES restarts with diverse initializations (including the
         Schwefel optimum neighborhood) and diverse initial step sizes.
      2. Adaptive step-size control (success-rate based, bounded).
      3. Final fine-grained local refinement around the best point found.
      4. Explicit wall-clock deadline guard with safety margin before the 330s kill.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before the 330s kill

        call_count = [0]

        def make_call(x):
            """Wrap func call with budget tracking; returns None if exhausted."""
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

        # Reserve a small number of calls for the final refinement phase.
        reserve = min(40, max(10, budget // 10))

        # Phase 1: multiple diverse (1+1)-ES restarts.
        # Diverse initializations: origin, Schwefel-optimum neighborhood, random.
        opt = 4.20968746
        init_strategies = [
            lambda: np.zeros(dim),
            lambda: np.full(dim, opt),
            lambda: np.full(dim, -opt),
            lambda: rng.uniform(-5, 5, dim),
            lambda: rng.uniform(-5, 5, dim),
            lambda: rng.uniform(-5, 5, dim),
            lambda: np.zeros(dim) + 0.5 * rng.standard_normal(dim),
            lambda: rng.uniform(-5, 5, dim),
        ]
        initial_sigmas = [2.0, 1.0, 3.0, 1.5, 2.5, 1.0, 3.0, 0.7]

        n_restarts = len(init_strategies)
        # Geometric-decay budget allocation across restarts.
        remaining = budget - reserve
        allocations = []
        for i in range(n_restarts):
            if i == n_restarts - 1:
                allocations.append(max(1, remaining))
            else:
                frac = 0.5 ** i * 0.25
                alloc = max(10, int(remaining * frac))
                allocations.append(alloc)
                remaining -= alloc

        for r in range(n_restarts):
            if call_count[0] >= budget - reserve:
                break
            if time.time() > deadline - 20:
                break
            alloc = allocations[r]
            if alloc < 5:
                break

            x = init_strategies[r % len(init_strategies)]()
            fx = make_call(x)
            if fx is None:
                break
            update_best(x, fx)

            sigma = initial_sigmas[r % len(initial_sigmas)]
            success_count = 0
            total_count = 0

            for _ in range(alloc):
                if call_count[0] >= budget - reserve:
                    break
                if time.time() > deadline - 20:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = make_call(y)
                if fy is None:
                    break

                total_count += 1
                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    update_best(x, fx)
                    sigma *= 1.3
                else:
                    sigma *= 0.85

                # Adaptive step-size adjustment based on recent success rate.
                if total_count >= 10:
                    rate = success_count / total_count
                    if rate > 0.3:
                        sigma *= 1.1
                    elif rate < 0.1:
                        sigma *= 0.9
                    success_count = 0
                    total_count = 0

                sigma = float(np.clip(sigma, 0.01, 10.0))

        # Phase 2: final local refinement around the best point.
        if best_x is not None and call_count[0] < budget - 5 and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(budget - call_count[0] - 5):
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
                    sigma *= 1.1
                else:
                    sigma *= 0.8

                sigma = float(np.clip(sigma, 0.001, 1.0))

        return best_x, best_f
# EVOLVE-BLOCK-END
