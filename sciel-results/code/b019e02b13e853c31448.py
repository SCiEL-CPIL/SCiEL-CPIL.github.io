import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """(1+1)-ES with adaptive sigma and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        calls = 1

        best_x = x.copy()
        best_fx = fx
        sigma = 3.0

        # Phase 1: (1+1)-ES for most of the budget
        # Reserve last 50 calls for Nelder-Mead polish
        polish_budget = 50
        es_budget = budget - polish_budget

        for _ in range(es_budget - 1):
            if time.time() >= deadline:
                break
            if calls >= budget:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
            else:
                sigma *= 0.9

            # Clamp sigma
            if sigma > 15.0:
                sigma = 15.0
            elif sigma < 1e-14:
                sigma = 1e-14

        # Phase 2: Nelder-Mead polish using remaining budget
        remaining = budget - calls
        if remaining > 2 and time.time() < deadline:
            # Wrap func to track calls and enforce deadline
            nm_calls = [0]
            
            def nm_func(z):
                if time.time() >= deadline:
                    raise Exception("timeout")
                if nm_calls[0] >= remaining:
                    raise Exception("budget exceeded")
                nm_calls[0] += 1
                return func(z)
            
            # Use Nelder-Mead from best point
            try:
                result = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': remaining, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                if result.fun < best_fx:
                    best_x = result.x
                    best_fx = result.fun
            except Exception:
                pass

        # Phase 3: Additional L-BFGS-B polish if budget remains
        remaining2 = budget - calls
        if remaining2 > 2 and time.time() < deadline:
            lb_calls = [0]
            
            def lb_func(z):
                if time.time() >= deadline:
                    raise Exception("timeout")
                if lb_calls[0] >= remaining2:
                    raise Exception("budget exceeded")
                lb_calls[0] += 1
                return func(z)
            
            try:
                bounds = [(-5, 5)] * dim
                result2 = minimize(
                    lb_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': remaining2, 'gtol': 1e-12, 'ftol': 1e-14}
                )
                calls += lb_calls[0]
                if result2.fun < best_fx:
                    best_x = result2.x
                    best_fx = result2.fun
            except Exception:
                pass

        return best_x, best_fx
