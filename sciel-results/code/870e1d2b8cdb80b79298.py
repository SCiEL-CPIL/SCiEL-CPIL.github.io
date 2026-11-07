"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-restart CMA-ES style optimizer with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin within 300s limit
        budget = self.budget
        dim = self.dim

        # Count function calls
        call_count = 0

        def wrapped_func(x):
            nonlocal call_count
            if call_count >= budget:
                return float('inf')
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = x.reshape(dim)
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            call_count += 1
            return float(val)

        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')

        # Phase 1: Multi-start with L-BFGS-B
        n_starts = 15
        for i in range(n_starts):
            if call_count >= budget or time.time() > deadline - 10:
                break

            if i == 0:
                x0 = np.zeros(dim)
            elif i == 1:
                x0 = np.ones(dim) * 3.0
            elif i == 2:
                x0 = -np.ones(dim) * 3.0
            else:
                x0 = rng.uniform(-4.5, 4.5, dim)

            try:
                result = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 200, 'maxfun': max(1, budget - call_count), 'gtol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

            if time.time() > deadline - 30:
                break

        if best_x is None:
            # Fallback: uniform random search
            for _ in range(budget - call_count):
                if time.time() > deadline - 5:
                    break
                x = rng.uniform(-5, 5, dim)
                fx = wrapped_func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Phase 2: Local refinement with Nelder-Mead around best point
        if best_x is not None and time.time() < deadline - 20 and call_count < budget:
            try:
                nm_result = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 100, 'maxfev': max(1, budget - call_count), 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if nm_result.fun < best_f:
                    best_f = nm_result.fun
                    best_x = nm_result.x.copy()
            except Exception:
                pass

        # Phase 3: If time remains, do a few more random restarts with L-BFGS-B
        while time.time() < deadline - 10 and call_count < budget:
            x0 = rng.uniform(-4.5, 4.5, dim)
            try:
                result = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 100, 'maxfun': max(1, budget - call_count), 'gtol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END
