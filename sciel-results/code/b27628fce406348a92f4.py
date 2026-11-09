"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """(1+1)-ES with 1/5 success rule, time-bounded, plus Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x, best_fx = x.copy(), fx
        sigma = 2.0
        calls = 1

        # Run (1+1)-ES, reserving some budget for polish
        # Stop ES when we've used ~80% of budget or are getting close to deadline
        es_max_calls = int(0.80 * self.budget)

        for i in range(es_max_calls - 1):
            if calls >= self.budget:
                break
            if time.time() > deadline - 5.0:
                break

            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.87

            if fx < best_fx:
                best_x, best_fx = x.copy(), fx

        # Nelder-Mead polish pass on the best point found
        remaining = self.budget - calls
        if remaining > 5 and time.time() < deadline - 2.0:
            try:
                result = minimize(
                    func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'maxfev': remaining,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                        'initial_simplex': np.array(
                            [best_x + 0.01 * np.eye(self.dim)[i] for i in range(self.dim)] +
                            [best_x]
                        )
                    }
                )
                if result.fun < best_fx:
                    best_x = result.x
                    best_fx = result.fun
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
