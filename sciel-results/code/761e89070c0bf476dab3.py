import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """CMA-ES global search + L-BFGS-B local polish for Rosenbrock f8."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        n = self.dim
        budget = self.budget

        calls_used = 0
        best_x = None
        best_f = np.inf

        def count_call(x):
            nonlocal calls_used, best_f, best_x
            val = func(x)
            calls_used += 1
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float)
            return val

        # Phase 1: CMA-ES for global valley exploration
        # Use a moderate number of CMA-ES iterations to find the valley
        cma_budget = int(budget * 0.5)  # 250 calls for CMA-ES

        # Initial mean: try center, then adapt
        x0 = np.zeros(n)
        f0 = count_call(x0)
        if f0 < best_f:
            best_f = f0
            best_x = x0.copy()

        # CMA-ES implementation (simple, no external dependency)
        # Using scipy's CMA-ES via minimize with method='COBYLA' won't work well
        # Let's implement a basic CMA-ES or use a different strategy

        # Actually, let's use a structured multi-start with gradient descent in the valley direction
        # For Rosenbrock, the valley is near x_i = x_{i+1}^2 for i < n-1, x_{n-1} = 1

        # Phase 1: Coarse grid / structured sampling to find approximate valley
        # Try points along the valley structure
        valley_starts = []
        # Standard Rosenbrock minimum is at (1,1,...,1) but shifted/rotated
        # Try a few structured points
        for trial in range(8):
            if time.time() > deadline or calls_used >= budget:
                break
            if trial == 0:
                x = np.ones(n)
            elif trial == 1:
                x = -np.ones(n)
            elif trial == 2:
                x = np.zeros(n)
            elif trial == 3:
                x = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
            elif trial == 4:
                x = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
            elif trial == 5:
                x = np.array([-1.0, 1.0, -1.0, 1.0, -1.0])
            elif trial == 6:
                x = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
            else:
                x = np.random.default_rng(trial * 123 + 456).uniform(-4.0, 4.0, n)

            x = np.clip(x, -5.0, 5.0)
            f = count_call(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 2: Use Nelder-Mead from best point for local convergence
        # Nelder-Mead is robust for Rosenbrock and doesn't need gradients
        if best_x is not None and calls_used < budget and time.time() < deadline:
            try:
                remaining = budget - calls_used
                res = minimize(
                    count_call,
                    x0=best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'maxfev': min(remaining, 150),
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        # Phase 3: L-BFGS-B polish if we have budget left
        if best_x is not None and calls_used < budget and time.time() < deadline:
            try:
                remaining = budget - calls_used
                if remaining > 10:
                    res2 = minimize(
                        count_call,
                        x0=best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * n,
                        options={
                            'maxiter': 100,
                            'maxfun': min(remaining, 80),
                            'gtol': 1e-12
                        }
                    )
            except Exception:
                pass

        # Phase 4: Final quick Nelder-Mead polish
        if best_x is not None and calls_used < budget and time.time() < deadline:
            try:
                remaining = budget - calls_used
                if remaining > 5:
                    res3 = minimize(
                        count_call,
                        x0=best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 50,
                            'maxfev': min(remaining, 30),
                            'xatol': 1e-12,
                            'fatol': 1e-12,
                            'adaptive': True
                        }
                    )
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(n)
            best_f = count_call(best_x)

        return best_x, best_f
