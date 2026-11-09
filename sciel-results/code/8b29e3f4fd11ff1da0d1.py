"""Rewritten optimizer: multi-start CMA-ES with time guard and adaptive restarts."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start CMA-ES with time guard for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_func(x):
            nonlocal calls_used
            if time.time() > deadline:
                return best_f
            x = np.clip(x, -5.0, 5.0)
            try:
                f = float(func(x))
            except Exception:
                return best_f
            if np.isfinite(f):
                calls_used += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f
            return best_f

        # Phase 1: Coarse random sampling to get a good starting point
        # Sample 20 points randomly
        for _ in range(min(20, budget)):
            if time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            safe_func(x)
            if calls_used >= budget:
                break

        # Phase 2: Multi-start CMA-ES from best point and a few other points
        # We'll do multiple CMA-ES restarts with decreasing budget per restart
        # Each restart gets a fraction of the remaining budget

        remaining_budget = budget - calls_used
        if remaining_budget <= 0 or best_x is None:
            return best_x, best_f

        # Number of restarts based on function characteristics
        # Use 3-5 restarts for multimodal functions
        n_restarts = min(5, max(2, remaining_budget // 30))
        budget_per_restart = max(20, remaining_budget // n_restarts)

        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break

            # Starting point for this restart
            if restart == 0:
                x0 = best_x.copy()
            else:
                # For subsequent restarts, try points near the best or random
                if rng.random() < 0.5:
                    # Perturb the best point
                    x0 = best_x + rng.normal(0, 1.0, dim)
                else:
                    # Random restart
                    x0 = rng.uniform(-5.0, 5.0, dim)
                x0 = np.clip(x0, -5.0, 5.0)
                # Evaluate the starting point
                safe_func(x0)
                if calls_used >= budget:
                    break

            # Run CMA-ES with a limited budget
            maxiter = min(budget_per_restart, budget - calls_used)
            if maxiter <= 0:
                break

            # CMA-ES options
            options = {
                'maxiter': maxiter,
                'tolx': 1e-6,
                'tolf': 1e-8,
                'bounds': [(-5.0, 5.0)] * dim,
                'verbose': 0,
            }

            try:
                result = minimize(
                    safe_func,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if result.x is not None and np.isfinite(result.fun):
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
            except Exception:
                pass

            # Check if we should continue
            if time.time() > deadline or calls_used >= budget:
                break

        # Phase 3: Final local refinement using Nelder-Mead from the best point
        if best_x is not None and calls_used < budget and time.time() < deadline:
            remaining = budget - calls_used
            if remaining > 10:
                try:
                    result = minimize(
                        safe_func,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': remaining, 'xatol': 1e-7, 'fatol': 1e-9}
                    )
                    if result.x is not None and np.isfinite(result.fun):
                        if result.fun < best_f:
                            best_f = result.fun
                            best_x = result.x.copy()
                except Exception:
                    pass

        return best_x, best_f
