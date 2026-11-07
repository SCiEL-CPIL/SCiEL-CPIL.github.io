"""Hybrid: Random sampling followed by SciPy Powell/Nelder-Mead local optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid: Random sampling followed by SciPy Powell/Nelder-Mead local optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 330s limit
        rng = np.random.default_rng(42)

        # Track best
        best_x = None
        best_f = np.inf
        calls = 0

        def try_eval(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Random sampling to find a good starting point
        # Discus is anisotropic, so random sampling in the box is a robust global search
        n_random = 50
        for _ in range(n_random):
            if calls >= budget:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            try_eval(x)
        
        # Evaluate origin as a fallback/anchor
        if best_x is None:
            x0 = np.zeros(dim)
            try_eval(x0)
        
        if best_x is None or calls >= budget:
            return best_x, best_f

        # Phase 2: Local optimization using SciPy
        # Powell's method is effective for anisotropic functions as it adapts search directions
        # We use a wrapper to count calls and stop if budget is exceeded
        # However, minimize() doesn't easily accept a dynamic budget stop.
        # We estimate maxiter based on remaining budget.
        # Powell uses ~dim*2 calls per iteration? Let's be conservative.
        
        remaining_budget = budget - calls
        if remaining_budget <= 0:
            return best_x, best_f

        # Use Nelder-Mead or Powell
        # Powell is often better for anisotropy, but can be slower per iteration.
        # Nelder-Mead is faster but can get stuck.
        # Let's try Powell first.
        
        # We need to handle the case where minimize exceeds budget.
        # We can wrap the function to raise an exception if budget is hit.
        class BudgetExceeded(Exception):
            pass

        def func_wrapper(x):
            nonlocal calls
            if calls >= budget:
                raise BudgetExceeded()
            return try_eval(x)

        try:
            # Powell method
            result = minimize(
                func_wrapper,
                best_x,
                method='Powell',
                options={'maxiter': 100, 'maxfev': remaining_budget} # maxfev is supported in newer scipy, but not always.
            )
        except BudgetExceeded:
            pass
        except Exception:
            # Fallback to Nelder-Mead if Powell fails or is not supported properly
            try:
                result = minimize(
                    func_wrapper,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100}
                )
            except BudgetExceeded:
                pass
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
