"""Optimized program: (1+1)-ES multi-start + L-BFGS-B polish with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start (1+1)-ES exploration + L-BFGS-B polish, deadline-guarded."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 295.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        calls = [0]

        def counted(x):
            calls[0] += 1
            return func(x)

        # Exploration: multi-start (1+1)-ES
        rng = np.random.default_rng(12345)
        # Allocate exploration budget: ~70% of total
        explore_budget = int(budget * 0.65)
        # Number of ES restarts
        n_restarts = 5
        per_restart = max(1, explore_budget // n_restarts)

        best_x = None
        best_f = np.inf

        for r in range(n_restarts):
            if calls[0] >= budget - 50:
                break
            if time.time() > deadline - 30:
                break

            # Random start, biased toward center for better initial coverage
            if r == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-5, 5, dim)
            fx = counted(x)
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            sigma = 1.5
            # (1+1)-ES with 1/5 success rule
            successes = 0
            attempts = 0
            for _ in range(per_restart):
                if calls[0] >= budget - 50:
                    break
                if time.time() > deadline - 30:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = counted(y)
                attempts += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    sigma *= 1.25
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    sigma *= 0.8
                # 1/5 success rule: check every 5 attempts
                if attempts % 5 == 0:
                    if successes / attempts < 0.2:
                        sigma *= 0.8
                    elif successes / attempts > 0.5:
                        sigma *= 1.25
                    successes = 0
                    attempts = 0

        # Polish: L-BFGS-B from best point
        if best_x is not None and calls[0] < budget - 10:
            remaining = budget - calls[0]
            # L-BFGS-B uses finite differences: each gradient eval costs (dim+1) func calls
            # So maxiter should be small
            maxiter_polish = min(remaining // (dim + 1), 20)
            if maxiter_polish > 0:
                try:
                    res = minimize(
                        counted,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={'maxiter': maxiter_polish, 'maxfun': remaining, 'ftol': 1e-12, 'gtol': 1e-12},
                    )
                    if res.fun < best_f:
                        best_f, best_x = res.fun, res.x.copy()
                except Exception:
                    pass

        # Final: if we have budget left, do a few more ES steps from best
        if best_x is not None and calls[0] < budget - 5:
            remaining = budget - calls[0]
            sigma = 0.5
            for _ in range(min(remaining, 20)):
                if time.time() > deadline:
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                fy = counted(y)
                if fy < best_f:
                    best_f, best_x = fy, y.copy()
                    sigma *= 1.3
                else:
                    sigma *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
