"""Rewritten optimizer: hybrid multi-start L-BFGS-B with CMA-ES-style global phase."""
import time
import numpy as np
from scipy.optimize import minimize as scipy_minimize


class Optimizer:
    """Hybrid optimizer: global exploration + local L-BFGS-B refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim

        # Track best
        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_call(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return None
            try:
                fx = float(func(np.asarray(x, dtype=float)))
            except Exception:
                return None
            call_count += 1
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = np.array(x, dtype=float)
            return fx

        # Phase 1: Global exploration with multiple random starts
        # Use a portion of budget for global search
        n_global = min(50, budget // 6)
        rng = np.random.default_rng(42)

        for i in range(n_global):
            if time.time() > deadline or call_count >= budget:
                break
            x0 = rng.uniform(-5, 5, dim)
            fx = safe_call(x0)
            if fx is None:
                continue
            # Quick local refinement from each start using Nelder-Mead (fast, no gradient)
            if time.time() > deadline or call_count >= budget - 5:
                continue
            try:
                res = scipy_minimize(
                    lambda x: safe_call(x),
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 5, 'fatol': 1e-10, 'xatol': 1e-10}
                )
                if hasattr(res, 'x') and res.x is not None:
                    fx2 = safe_call(res.x)
            except Exception:
                pass

        # Phase 2: Local refinement with L-BFGS-B from best point and neighbors
        if best_x is not None and call_count < budget and time.time() < deadline:
            # L-BFGS-B from the best point
            for attempt in range(3):
                if time.time() > deadline or call_count >= budget:
                    break
                x_start = best_x.copy()
                if attempt > 0:
                    x_start = x_start + rng.normal(0, 0.1, dim)
                    x_start = np.clip(x_start, -5, 5)

                try:
                    res = scipy_minimize(
                        lambda x: safe_call(x),
                        x_start,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-10}
                    )
                    if hasattr(res, 'x') and res.x is not None:
                        safe_call(res.x)
                except Exception:
                    pass

        # Phase 3: Final polish with tighter L-BFGS-B
        if best_x is not None and call_count < budget and time.time() < deadline:
            try:
                res = scipy_minimize(
                    lambda x: safe_call(x),
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-18, 'gtol': 1e-12}
                )
                if hasattr(res, 'x') and res.x is not None:
                    safe_call(res.x)
            except Exception:
                pass

        return best_x, best_f
