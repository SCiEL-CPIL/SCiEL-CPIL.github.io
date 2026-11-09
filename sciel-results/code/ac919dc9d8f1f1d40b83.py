"""Optimized solver for BBOB f9 (rotated Rosenbrock) using CMA-ES + L-BFGS-B hybrid."""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Hybrid CMA-ES (global) + L-BFGS-B (local polish) optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Wrapper that clips to box and counts calls
        call_count = [0]
        best_x = [None]
        best_f = [np.inf]

        def counted_func(x):
            call_count[0] += 1
            if call_count[0] > budget:
                return best_f[0]
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            if f < best_f[0]:
                best_f[0] = f
                best_x[0] = x_clipped.copy()
            return f

        rng = np.random.default_rng(42)

        # Track best solution
        overall_best_x = None
        overall_best_f = np.inf

        # Phase 1: CMA-ES for global valley detection
        # CMA-ES adapts covariance to the rotated landscape, efficiently
        # navigating the narrow Rosenbrock valley.
        cma_budget = int(budget * 0.6)  # 60% of budget for global search
        cma_max_calls = cma_budget

        try:
            from cma import CMA, CMAEvolutionStrategy
        except ImportError:
            CMAEvolutionStrategy = None

        if CMAEvolutionStrategy is not None:
            # CMA-ES parameters tuned for rotated Rosenbrock in [-5,5]^5
            x0 = np.zeros(dim)
            sigma0 = 2.0  # initial step size
            opts = {
                'maxiter': 500,
                'bounds': [(-5.0, 5.0)] * dim,
                'seed': 42,
                'verbose': -9,
                'tolopt': 1e-12,
                'tolx': 1e-10,
                'popsize': max(4, dim + 3),
            }

            # Use a wrapper that respects our call budget
            cma_call_count = [0]

            def cma_func(x):
                cma_call_count[0] += 1
                if cma_call_count[0] > cma_max_calls or call_count[0] >= budget:
                    return best_f[0]
                return counted_func(x)

            try:
                es = CMAEvolutionStrategy(x0, sigma0, opts)
                while not es.stop() and cma_call_count[0] < cma_max_calls \
                        and call_count[0] < budget and time.time() < deadline - 60.0:
                    x = es.ask()
                    f = cma_func(x)
                    es.tell(x, f)
                    if f < overall_best_f:
                        overall_best_f = f
                        overall_best_x = np.clip(x, -5, 5).copy()

                # Get best point from CMA-ES
                if es.result.xbest is not None:
                    xb = np.clip(es.result.xbest, -5, 5)
                    if es.result.fbest < overall_best_f:
                        overall_best_f = es.result.fbest
                        overall_best_x = xb.copy()
            except Exception:
                pass
        else:
            # Fallback: multi-start L-BFGS-B if cma not available
            starts = []
            starts.append(np.zeros(dim))
            for _ in range(20):
                starts.append(rng.normal(0, 1.5, dim))
            for scale in [0.5, 1.0, 2.0, 3.0]:
                starts.append(np.full(dim, scale))
                starts.append(np.full(dim, -scale))
            for _ in range(15):
                starts.append(rng.uniform(-4, 4, dim))

            bounds = [(-5.0, 5.0)] * dim
            for start in starts:
                if call_count[0] >= cma_max_calls or time.time() > deadline - 60.0:
                    break
                start = np.clip(start, -5, 5)
                try:
                    res = minimize(
                        counted_func, start,
                        method='L-BFGS-B', bounds=bounds,
                        options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    if res.fun < overall_best_f:
                        overall_best_f = res.fun
                        overall_best_x = np.clip(res.x, -5, 5).copy()
                except Exception:
                    continue

        # Phase 2: High-precision L-BFGS-B polishing
        if overall_best_x is not None and call_count[0] < budget and time.time() < deadline - 20.0:
            bounds = [(-5.0, 5.0)] * dim
            # Multiple refinement passes with tightening tolerances
            for polish_pass in range(4):
                if call_count[0] >= budget or time.time() > deadline - 20.0:
                    break
                remaining = budget - call_count[0]
                if remaining < 10:
                    break

                ftol = 1e-14
                gtol = 1e-13
                maxiter = min(100, max(20, remaining // 2))

                try:
                    res = minimize(
                        counted_func,
                        overall_best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
                    )
                    if res.fun < overall_best_f:
                        overall_best_f = res.fun
                        overall_best_x = np.clip(res.x, -5, 5).copy()
                except Exception:
                    break

        # Phase 3: Final ultra-precise polish if budget remains
        if overall_best_x is not None and call_count[0] < budget - 5 and time.time() < deadline - 5.0:
            remaining = budget - call_count[0]
            if remaining >= 15:
                bounds = [(-5.0, 5.0)] * dim
                try:
                    res = minimize(
                        counted_func,
                        overall_best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': min(60, remaining - 5), 'ftol': 1e-15, 'gtol': 1e-14}
                    )
                    if res.fun < overall_best_f:
                        overall_best_f = res.fun
                        overall_best_x = np.clip(res.x, -5, 5).copy()
                except Exception:
                    pass

        # Return the best solution found
        if overall_best_x is None:
            if best_x[0] is not None:
                return best_x[0], best_f[0]
            return np.zeros(dim), 0.0

        return overall_best_x, overall_best_f
