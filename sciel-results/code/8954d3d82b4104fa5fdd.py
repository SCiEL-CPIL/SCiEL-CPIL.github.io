"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Combined LHS + Powell optimizer for BBOB f1 (Sphere)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        n = self.dim
        budget = self.budget
        calls = [0]
        best_x = None
        best_f = float('inf')

        def wrapped_func(x):
            nonlocal best_x, best_f
            if calls[0] >= budget:
                return float('inf')
            calls[0] += 1
            val = func(np.clip(x, -5, 5))
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float)
            return val

        # Phase 1: Latin Hypercube Sampling for broad exploration
        n_lhs = 20
        if budget > n_lhs:
            # Generate LHS points
            lhs_points = []
            for i in range(n_lhs):
                point = []
                for d in range(n):
                    low = -5 + (10 * i / n_lhs)
                    high = -5 + (10 * (i + 1) / n_lhs)
                    point.append(np.random.uniform(low, high))
                lhs_points.append(point)

            lhs_points = np.array(lhs_points)

            # Evaluate all LHS points
            for x in lhs_points:
                if calls[0] >= budget or time.time() > deadline:
                    break
                wrapped_func(x)

        if best_x is None:
            # Fallback: evaluate origin
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls[0] = 1

        if calls[0] >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Powell's Method for local refinement
        remaining = budget - calls[0]
        if remaining > 10:
            try:
                res = minimize(wrapped_func, best_x, method='Powell',
                               bounds=[(-5, 5)] * n,
                               options={'maxiter': min(500, remaining // 2),
                                        'xatol': 1e-8, 'fatol': 1e-12})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.array(res.x, dtype=float)
            except Exception:
                pass

        if calls[0] >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 3: Fine-tuning with Nelder-Mead for final precision
        remaining = budget - calls[0]
        if remaining > 5:
            try:
                res = minimize(wrapped_func, best_x, method='Nelder-Mead',
                               options={'maxiter': min(200, remaining),
                                        'xatol': 1e-10, 'fatol': 1e-14,
                                        'adaptive': True})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.array(res.x, dtype=float)
            except Exception:
                pass

        # Ensure best_x is within bounds
        best_x = np.clip(best_x, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
