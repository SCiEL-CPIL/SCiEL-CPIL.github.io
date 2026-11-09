"""Hybrid (1+1)-ES with multi-start Nelder-Mead polish for BBOB f9."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """(1+1)-ES with 1/5 success rule, followed by multi-start Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        rng = np.random.default_rng(0)

        # Phase 1: (1+1)-ES with 1/5 success rule
        # Reserve some budget for polishing
        es_budget = int(self.budget * 0.80)
        polish_budget = self.budget - es_budget

        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        sigma = 2.0
        calls_made = 1

        for _ in range(es_budget - 1):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_made += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma *= 0.87

        # Phase 2: Multi-start Nelder-Mead polish
        # Use multiple starting points to avoid local minima
        if polish_budget > 5 and time.time() < deadline:
            # Create multiple starting points: best_x plus perturbations
            start_points = [best_x.copy()]
            # Add a few perturbed versions of best_x
            for i in range(2):
                perturbation = 0.3 * rng.standard_normal(self.dim)
                perturbed = np.clip(best_x + perturbation, -5, 5)
                start_points.append(perturbed)
            # Add one random point
            start_points.append(rng.uniform(-5, 5, self.dim))

            # Split polish budget among starting points
            n_starts = len(start_points)
            budget_per_start = max(3, polish_budget // n_starts)

            for sp in start_points:
                if time.time() > deadline:
                    break
                if polish_budget <= 0:
                    break

                nm_calls = [0]
                nm_limit = min(budget_per_start, polish_budget)

                def nm_func(x_in, _calls=nm_calls, _limit=nm_limit):
                    if time.time() > deadline:
                        return np.inf
                    if _calls[0] >= _limit:
                        return np.inf
                    _calls[0] += 1
                    try:
                        return func(x_in)
                    except Exception:
                        return np.inf

                try:
                    res = minimize(
                        nm_func,
                        sp,
                        method='Nelder-Mead',
                        options={
                            'maxiter': nm_limit * 3,
                            'xatol': 1e-6,
                            'fatol': 1e-6,
                            'adaptive': True,
                            'initial_simplex_size': 0.3
                        }
                    )
                    polish_budget -= nm_calls[0]
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    polish_budget -= nm_calls[0]

        return best_x, best_f
# EVOLVE-BLOCK-END
