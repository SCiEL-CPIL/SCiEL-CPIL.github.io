"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with 1/5 success rule, restarts, and polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Multi-start: diverse initial points to cover Rastrigin basins
        starts = [
            np.zeros(dim),
            rng.uniform(-5, 5, dim),
            rng.uniform(-5, 5, dim),
            rng.uniform(-2, 2, dim),
            rng.uniform(-5, 5, dim),
            np.ones(dim) * 3.0,
        ]

        best_x = None
        best_f = np.inf
        calls_made = 0

        def run_es(x0, max_calls, sigma_init=2.0):
            nonlocal calls_made, best_x, best_f
            x = x0.copy()
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f, best_x = fx, x.copy()
            sigma = sigma_init
            for _ in range(max_calls):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    sigma *= 0.85
                    sigma = max(sigma, 1e-6)
            return x, fx

        # Phase 1: Multi-start exploration
        # Split budget among restarts, with more budget for later (better) starts
        # 6 restarts, allocate budget proportionally
        remaining = budget
        restart_budgets = []
        for i in range(len(starts)):
            if i < len(starts) - 1:
                rb = max(10, remaining // (len(starts) - i))
                restart_budgets.append(rb)
                remaining -= rb
            else:
                restart_budgets.append(max(10, remaining))

        for i, (start, rb) in enumerate(zip(starts, restart_budgets)):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break
            run_es(start, rb, sigma_init=2.0)

        # Phase 2: Polish around best found
        if best_x is not None and calls_made < budget and time.time() < deadline:
            polish_budget = min(budget - calls_made, 100)
            sigma_polish = 0.1
            x = best_x.copy()
            fx = best_f
            for _ in range(polish_budget):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break
                y = np.clip(x + sigma_polish * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma_polish *= 1.2
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    sigma_polish *= 0.8
                    sigma_polish = max(sigma_polish, 1e-8)

        # Phase 3: Final fine polish
        if best_x is not None and calls_made < budget and time.time() < deadline:
            final_budget = budget - calls_made
            sigma_final = 0.01
            x = best_x.copy()
            fx = best_f
            for _ in range(final_budget):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break
                y = np.clip(x + sigma_final * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma_final *= 1.1
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    sigma_final *= 0.9
                    sigma_final = max(sigma_final, 1e-10)

        return best_x, best_f
# EVOLVE-BLOCK-END
