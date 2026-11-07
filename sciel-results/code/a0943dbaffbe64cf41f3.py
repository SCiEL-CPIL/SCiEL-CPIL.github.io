"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, NelderMead


class Optimizer:
    """Multi-start local optimization for Rosenbrock f8."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        # Wrap func to count calls and clip
        call_count = [0]
        max_calls = self.budget

        def wrapped(x):
            if call_count[0] >= max_calls:
                return np.inf
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            call_count[0] += 1
            return float(val)

        dim = self.dim

        # Generate diverse starting points
        starts = []
        # Center
        starts.append(np.zeros(dim))
        # Axis-aligned positive
        for i in range(dim):
            p = np.zeros(dim)
            p[i] = 3.0
            starts.append(p.copy())
        # Axis-aligned negative
        for i in range(dim):
            p = np.zeros(dim)
            p[i] = -3.0
            starts.append(p.copy())
        # Random points
        rng = np.random.default_rng(42)
        for _ in range(6):
            starts.append(rng.uniform(-4, 4, dim))

        best_x = None
        best_f = np.inf

        for start in starts:
            if call_count[0] >= max_calls:
                break
            if time.time() > deadline - 5.0:
                break

            # Try Nelder-Mead first (derivative-free, robust)
            try:
                res = minimize(
                    wrapped,
                    start,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

            if call_count[0] >= max_calls:
                break
            if time.time() > deadline - 5.0:
                break

            # Try SLSQP for potential quadratic convergence near optimum
            try:
                bounds = [(-5.0, 5.0)] * dim
                res2 = minimize(
                    wrapped,
                    best_x if best_x is not None else start,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12}
                )
                if res2.fun < best_f:
                    best_f = res2.fun
                    best_x = res2.x.copy()
            except Exception:
                pass

            if call_count[0] >= max_calls:
                break
            if time.time() > deadline - 5.0:
                break

            # Final high-precision polish with Nelder-Mead if budget remains
            if best_x is not None:
                try:
                    res3 = minimize(
                        wrapped,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                    )
                    if res3.fun < best_f:
                        best_f = res3.fun
                        best_x = res3.x.copy()
                except Exception:
                    pass

        # Ensure we've made at least one call
        if call_count[0] == 0:
            x0 = np.zeros(dim)
            wrapped(x0)

        if best_x is None:
            best_x = np.zeros(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
