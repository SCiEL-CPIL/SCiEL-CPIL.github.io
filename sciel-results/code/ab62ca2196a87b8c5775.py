"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid LHS + CMA-ES + Nelder-Mead for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _latin_hypercube(self, rng, n_samples, dim):
        """Generate n_samples points via Latin Hypercube Sampling in [-5, 5]^dim."""
        points = np.zeros((n_samples, dim))
        for d in range(dim):
            perm = rng.permutation(n_samples)
            points[:, d] = (perm + 0.5) / n_samples * 10.0 - 5.0
        return points

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 280.0

        calls = 0

        def safe_func(x):
            nonlocal calls
            if time.time() > deadline:
                raise TimeoutError()
            if calls >= budget:
                raise TimeoutError()
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            calls += 1
            return func(x)

        # Phase 1: LHS initialization - 25 points for broad exploration
        n_lhs = min(25, budget // 10)
        lhs_points = self._latin_hypercube(rng, n_lhs, dim)
        lhs_values = np.array([safe_func(p) for p in lhs_points])
        best_idx = np.argmin(lhs_values)
        x_best = lhs_points[best_idx].copy()
        f_best = lhs_values[best_idx]

        if time.time() > deadline or calls >= budget:
            return x_best, f_best

        # Phase 2: CMA-ES from the best LHS point
        # Use scipy's CMA-ES via minimize with method='COBYLA' as a fallback
        # Actually, let's use a manual (1+1)-ES with better sigma adaptation
        # for more control over the budget

        # Let's use a more structured approach:
        # Phase 2a: (1+1)-ES with 1/5 success rule for ~300 calls
        # Phase 2b: Nelder-Mead polish for remaining calls

        remaining = budget - calls
        es_budget = min(300, remaining)
        nm_budget = max(50, remaining - es_budget)

        # (1+1)-ES phase
        x = x_best.copy()
        fx = f_best
        sigma = 1.0
        min_sigma = 0.001
        max_sigma = 5.0
        success_window = []

        for _ in range(es_budget):
            if time.time() > deadline or calls >= budget:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = safe_func(y)

            if fy < fx:
                x, fx = y, fy
                success_window.append(1)
            else:
                success_window.append(0)

            # Keep only last 10 evaluations for success rate
            if len(success_window) > 10:
                success_window.pop(0)

            # 1/5 success rule
            if len(success_window) >= 10:
                success_rate = sum(success_window) / len(success_window)
                if success_rate > 0.2:
                    sigma = min(sigma * 1.5, max_sigma)
                elif success_rate < 0.2:
                    sigma = max(sigma * 0.8, min_sigma)

        if time.time() > deadline or calls >= budget:
            return x, fx

        # Phase 3: Nelder-Mead polish
        nm_maxiter = max(1, (budget - calls) // (dim + 1))
        nm_maxiter = min(nm_maxiter, 200)

        try:
            result = minimize(
                safe_func,
                x,
                method='Nelder-Mead',
                options={
                    'maxiter': nm_maxiter,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            if result.fun < fx:
                x, fx = result.x, result.fun
        except Exception:
            pass

        return x, fx
# EVOLVE-BLOCK-END
