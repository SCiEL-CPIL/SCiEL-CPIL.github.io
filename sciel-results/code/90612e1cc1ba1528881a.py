"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """(1+1)-ES with adaptive sigma + Nelder-Mead polish, starting at origin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 280.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        calls = 0

        def _eval(x):
            nonlocal calls
            x = np.clip(x, -5, 5)
            calls += 1
            return func(x)

        # Start at origin - f14 minimum is typically near 0 after hidden transformations
        x = np.zeros(dim)
        fx = _eval(x)
        best_x = x.copy()
        best_f = fx
        sigma = 1.0

        # Main (1+1)-ES loop with conservative sigma adaptation
        es_budget = min(budget - 10, 400)  # leave room for NM polish
        for i in range(es_budget):
            if time.time() > deadline or calls >= budget:
                break
            y = x + sigma * rng.standard_normal(dim)
            fy = _eval(y)
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma *= 0.9
            sigma = np.clip(sigma, 1e-6, 10.0)

        # Nelder-Mead polish to tighten convergence near the optimum
        try:
            from scipy.optimize import minimize
            # Track calls in the wrapper
            nm_calls = [0]
            def nm_func(z):
                nm_calls[0] += 1
                return func(np.clip(z, -5, 5))

            # Use remaining budget for NM
            remaining = budget - calls
            maxiter = max(10, min(200, remaining // 2))
            res = minimize(
                nm_func,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-14,
                         'adaptive': True},
            )
            if res.fun < best_f:
                best_x = np.clip(res.x, -5, 5)
                best_f = res.fun
        except Exception:
            pass

        # Second polish pass with even tighter tolerances if budget remains
        remaining = budget - calls
        if remaining > 20:
            try:
                from scipy.optimize import minimize
                nm_calls2 = [0]
                def nm_func2(z):
                    nm_calls2[0] += 1
                    return func(np.clip(z, -5, 5))
                maxiter2 = max(5, min(100, remaining // 3))
                res2 = minimize(
                    nm_func2,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': maxiter2, 'xatol': 1e-14, 'fatol': 1e-16,
                             'adaptive': True},
                )
                if res2.fun < best_f:
                    best_x = np.clip(res2.x, -5, 5)
                    best_f = res2.fun
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
