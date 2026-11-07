"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start L-BFGS-B for Rosenbrock f8."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        rng = np.random.default_rng(42)
        n = self.dim

        # Diverse starting points: center, corners, random
        starts = [np.zeros(n)]
        for _ in range(4):
            starts.append(rng.uniform(-4.5, 4.5, n))
        # Add some structured points
        starts.append(np.array([1.0] * n))
        starts.append(np.array([-1.0] * n))

        best_x = None
        best_f = np.inf
        calls_used = 0

        for start in starts:
            if time.time() > deadline:
                break
            if calls_used >= self.budget:
                break

            # Run L-BFGS-B from this start
            try:
                res = minimize(
                    func,
                    x0=start,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * n,
                    options={'maxiter': 200, 'maxfun': 100000, 'gtol': 1e-8}
                )
                calls_used += res.nfev
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # If we have budget remaining, do a quick local polish from best
        if best_x is not None and calls_used < self.budget and time.time() < deadline:
            try:
                res2 = minimize(
                    func,
                    x0=best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * n,
                    options={'maxiter': 100, 'maxfun': 100000, 'gtol': 1e-10}
                )
                calls_used += res2.nfev
                if res2.fun < best_f:
                    best_f = res2.fun
                    best_x = res2.x.copy()
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
