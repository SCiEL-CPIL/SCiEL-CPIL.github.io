"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """(1+1)-ES with restarts and L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        best_x = None
        best_f = np.inf

        # We'll do multiple restarts of (1+1)-ES, each with a portion of the budget
        # Then polish the best candidate with L-BFGS-B

        # Strategy: Use ~70% of budget for ES exploration, ~30% for polish
        es_budget = int(self.budget * 0.7)
        polish_budget = self.budget - es_budget

        # Multiple restarts to escape local optima in bi-Rastrigin landscape
        n_restarts = 4
        budget_per_restart = es_budget // n_restarts

        rng = np.random.default_rng(42)

        for restart in range(n_restarts):
            if time.time() > deadline - 30:
                break

            # Initialize: first restart near origin, others random
            if restart == 0:
                x = np.zeros(self.dim)
            else:
                x = rng.uniform(-5, 5, self.dim)

            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 2.0
            calls_in_restart = 0
            max_calls = budget_per_restart

            for _ in range(max_calls):
                if time.time() > deadline - 30:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_in_restart += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

                    # Adaptive restart: if sigma gets too small, break early
                    if sigma < 0.01:
                        break

        # Polish the best candidate with L-BFGS-B
        if best_x is not None and time.time() < deadline - 10:
            try:
                def wrapper(x):
                    if time.time() > deadline - 5:
                        return np.inf
                    return func(np.clip(x, -5, 5))

                res = minimize(
                    wrapper,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 200, 'maxfun': polish_budget, 'gtol': 1e-8}
                )

                if res.success or res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Final check
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
