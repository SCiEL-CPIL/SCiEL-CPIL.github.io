"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size and final L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x, best_fx = x.copy(), fx

        sigma = 2.0
        calls = 1

        # Phase 1: (1+1)-ES with 1/5 success rule
        for i in range(self.budget - 1):
            if time.time() > deadline - 5.0:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
            else:
                sigma *= 0.87

        # Phase 2: Final polish with L-BFGS-B if we have budget left
        if calls < self.budget and time.time() < deadline - 2.0:
            try:
                from scipy.optimize import minimize
                # Use the best point found so far as starting point
                res = minimize(
                    func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 100, 'maxfun': self.budget - calls - 1}
                )
                calls += res.nfev
                if res.fun < best_fx:
                    best_x, best_fx = res.x.copy(), res.fun
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
