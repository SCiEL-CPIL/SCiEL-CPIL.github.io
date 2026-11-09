"""Multi-start adaptive (1+1)-ES with Sobol initialization for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Multi-start adaptive (1+1)-ES with Sobol initialization for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        calls_made = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global Exploration using Sobol Sequence
        n_init = 40
        sampler = qmc.Sobol(d=dim, scramble=True)
        sobol_points = sampler.random(n=n_init)
        init_points = sobol_points * 10.0 - 5.0

        init_results = []
        for p in init_points:
            if calls_made >= budget or time.time() > deadline:
                break
            f = evaluate(p)
            init_results.append((p, f))

        if not init_results:
            return best_x, best_f

        init_results.sort(key=lambda x: x[1])

        remaining_budget = budget - calls_made
        if remaining_budget <= 0:
            return best_x, best_f

        # Phase 2: Multi-Start Local Refinement using Adaptive (1+1)-ES
        max_candidates = min(5, len(init_results))
        alloc = [0.45, 0.25, 0.15, 0.10, 0.05]
        budgets = []
        for i in range(max_candidates):
            if i < len(alloc):
                budgets.append(int(remaining_budget * alloc[i]))
            else:
                budgets.append(10)

        for i in range(max_candidates):
            if time.time() > deadline or calls_made >= budget:
                break

            x0, f0 = init_results[i]
            local_budget = budgets[i]

            x = x0.copy()
            fx = f0
            sigma = 1.0
            stagnation = 0
            max_stagnation = 20
            es_calls = 0

            while es_calls < local_budget:
                if time.time() > deadline or calls_made >= budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = evaluate(y)
                es_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.25
                    stagnation = 0
                else:
                    sigma *= 0.8
                    stagnation += 1

                sigma = np.clip(sigma, 1e-6, 5.0)

                if stagnation >= max_stagnation:
                    break

        # Phase 3: Final Polish
        if time.time() < deadline and calls_made < budget and best_x is not None:
            x_polish = best_x.copy()
            f_polish = best_f
            sigma_p = 0.1
            max_polish = min(20, budget - calls_made)

            for _ in range(max_polish):
                if time.time() > deadline or calls_made >= budget:
                    break
                y = np.clip(x_polish + sigma_p * rng.standard_normal(dim), -5.0, 5.0)
                fy = evaluate(y)
                if fy < f_polish:
                    x_polish = y
                    f_polish = fy
                    sigma_p *= 0.9
                else:
                    sigma_p *= 0.95

        return best_x, best_f
# EVOLVE-BLOCK-END
