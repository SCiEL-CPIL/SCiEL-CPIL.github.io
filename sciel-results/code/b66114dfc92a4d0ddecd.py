"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Portfolio optimizer for BBOB f18 (Schaffers F7 ill-conditioned).
    Strategy:
      - Random sampling to seed the portfolio.
      - Multiple (1+1)-ES runs from diverse seeds with tuned adaptive step size.
      - Final L-BFGS-B polish from the global best.
      - Keep a global best across all runs.
      - Explicit wall-clock deadline guard.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 320.0
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(12345)

        calls = 0

        def safe_call(x):
            nonlocal calls
            calls += 1
            return func(x)

        # ---- Phase 1: random sampling ----
        n_samples = min(20, budget // 5)
        samples = []
        for _ in range(n_samples):
            if calls >= budget or time.time() >= deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            samples.append((fx, x))

        if not samples:
            return 0.0

        samples.sort(key=lambda t: t[0])
        best_fx, best_x = samples[0]

        # ---- Phase 2: multiple (1+1)-ES runs ----
        remaining = budget - calls
        if remaining <= 0 or time.time() >= deadline:
            return best_fx

        polish_budget = max(20, int(remaining * 0.2))
        search_budget = remaining - polish_budget

        # Select diverse seeds
        seeds = []
        for fx, x in samples[:10]:
            if all(np.linalg.norm(x - sx) > 0.5 for _, sx in seeds):
                seeds.append((fx, x))
        if not seeds:
            seeds = [samples[0]]

        n_runs = len(seeds)
        per_run_budget = max(1, search_budget // n_runs)

        for fx0, x0 in seeds:
            x = x0.copy()
            fx = fx0
            sigma = 1.5
            successes = 0
            total = 0
            for _ in range(per_run_budget):
                if calls >= budget or time.time() >= deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                total += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    sigma *= 1.15
                    if fx < best_fx:
                        best_fx, best_x = fx, x
                else:
                    sigma *= 0.9
                if total >= 1 + dim:
                    rate = successes / total
                    if rate > 0.35:
                        sigma *= 1.25
                    elif rate < 0.1:
                        sigma *= 0.6
                    successes = 0
                    total = 0
                sigma = np.clip(sigma, 1e-4, 8.0)

            if time.time() >= deadline or calls >= budget:
                break

        # ---- Phase 3: L-BFGS-B polish ----
        if calls < budget and time.time() < deadline:
            x_start = best_x.copy()

            polish_calls = [0]
            def polish_func(x):
                polish_calls[0] += 1
                return func(np.clip(x, -5, 5))

            try:
                res = minimize(polish_func, x_start, method='L-BFGS-B',
                               bounds=[(-5, 5)] * dim,
                               options={'maxiter': 50, 'maxfun': budget - calls - 1})
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_fx
# EVOLVE-BLOCK-END
