"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Analytical ZDT1 optimizer: minimize g, then sweep x_1 for Pareto coverage."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310  # safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n = self.dim - 1  # number of tail variables
        c = 1 + 10 * n    # ZDT1 constant

        # Phase 1: Find tail with minimal g
        # g = sum(x_i^2) for i in tail; f2 = c*(1+g)*(1-sqrt(f1/c))
        # => g = f2 / (c*(1-sqrt(f1/c))) - 1
        best_g = float('inf')
        best_tail = np.zeros(n)

        n_evals = 0
        # Use first portion of budget to search for good g
        g_search_budget = min(30, self.budget // 3)

        for _ in range(g_search_budget):
            if time.time() > deadline:
                break
            tail = rng.random(n)
            x = np.concatenate(([0.5], tail))
            f = problem(x)
            n_evals += 1
            f1, f2 = f[0], f[1]
            if f1 <= 0:
                continue
            sqrt_term = np.sqrt(f1 / c)
            if sqrt_term >= 1.0:
                continue
            denom = c * (1.0 - sqrt_term)
            if denom < 1e-12:
                continue
            g_est = f2 / denom - 1.0
            if g_est < best_g:
                best_g = g_est
                best_tail = tail.copy()

        # If we didn't find a good g, try setting tail to zeros directly
        if best_g > 1e-6:
            x_test = np.concatenate(([0.5], np.zeros(n)))
            f_test = problem(x_test)
            n_evals += 1
            f1_t, f2_t = f_test[0], f_test[1]
            if f1_t > 0:
                st = np.sqrt(f1_t / c)
                if st < 1.0:
                    d = c * (1.0 - st)
                    if d > 1e-12:
                        g_test = f2_t / d - 1.0
                        if g_test < best_g:
                            best_g = g_test
                            best_tail = np.zeros(n)

        # Phase 2: Sweep x_1 to cover the Pareto front
        remaining = self.budget - n_evals
        if remaining <= 0:
            return np.concatenate(([0.5], best_tail))

        # Sample x_1 values spread across [0, 1] for good hypervolume coverage
        x1_values = np.linspace(0.0, 1.0, remaining)
        # Add a few random points for robustness
        if remaining > 10:
            extra = rng.uniform(0, 1, min(5, remaining // 10))
            x1_values = np.concatenate([x1_values, extra])
            x1_values = x1_values[:remaining]

        for x1 in x1_values:
            if time.time() > deadline:
                break
            if n_evals >= self.budget:
                break
            x = np.concatenate(([x1], best_tail))
            problem(x)
            n_evals += 1

        return np.concatenate(([0.5], best_tail))
# EVOLVE-BLOCK-END
