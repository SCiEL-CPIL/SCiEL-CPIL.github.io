"""Optimizer for BBOB-style Schwefel f20 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid Nelder-Mead + (1+1)-ES with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        call_count = 0

        def eval_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget:
                return best_f
            fx = func(np.asarray(x, dtype=np.float64))
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = np.array(x, copy=True)
            return fx

        # Phase 1: Diversified Nelder-Mead from multiple starting points
        # Use structured + random starting points for basin coverage
        n_starts = 6
        start_points = []

        # Center
        start_points.append(np.zeros(self.dim))
        # Axis-aligned points
        for i in range(self.dim - 1):
            p = np.zeros(self.dim)
            p[i] = 4.0
            start_points.append(p)
        # Random points
        while len(start_points) < n_starts:
            p = rng.uniform(-4, 4, self.dim)
            if all(np.linalg.norm(p - q) > 1.0 for q in start_points):
                start_points.append(p)

        # Allocate budget: ~60% to NM, ~40% to ES polish
        nm_budget = int(self.budget * 0.6)
        es_budget = self.budget - nm_budget

        # Run Nelder-Mead from each start, tracking global best
        nm_calls = 0
        for sp in start_points:
            if call_count >= self.budget:
                break
            if time.time() > deadline:
                break

            # Limit calls per NM run
            max_nm_calls = max(10, nm_budget // n_starts)
            nm_calls_this = [0]

            def nm_obj(x):
                nonlocal nm_calls_this
                if nm_calls_this[0] >= max_nm_calls:
                    return best_f
                nm_calls_this[0] += 1
                return eval_func(x)

            try:
                res = minimize(
                    nm_obj,
                    sp,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_nm_calls,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'adaptive': True,
                        'initial_simplex': None,
                    }
                )
            except Exception:
                pass

        # Phase 2: (1+1)-ES polish from best point found
        if best_x is not None and call_count < self.budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5

            for _ in range(self.budget - call_count):
                if time.time() > deadline:
                    break
                if call_count >= self.budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = eval_func(y)

                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.2
                    if sigma > 3.0:
                        sigma = 3.0
                else:
                    sigma *= 0.9
                    if sigma < 0.01:
                        # Try a large jump to escape local minimum
                        jump = rng.uniform(-2, 2, self.dim)
                        y2 = np.clip(x + jump, -5, 5)
                        fy2 = eval_func(y2)
                        if fy2 < fx:
                            x = y2
                            fx = fy2
                            sigma = 1.0
                        else:
                            sigma = max(sigma * 0.8, 0.005)

        # Final: if we still have budget, do a few more random restarts
        if call_count < self.budget and best_x is not None:
            remaining = self.budget - call_count
            for _ in range(min(remaining, 20)):
                if time.time() > deadline:
                    break
                if call_count >= self.budget:
                    break
                # Small perturbation of best
                y = np.clip(best_x + 0.1 * rng.standard_normal(self.dim), -5, 5)
                fy = eval_func(y)

        return best_x, best_f
# EVOLVE-BLOCK-END
