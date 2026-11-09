"""Rewritten optimizer for BBOB f8 (Rosenbrock) with hidden transformations."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-restart Powell's method with budget-aware time management."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        # Wrap func to count calls and enforce budget
        call_count = [0]
        max_calls = self.budget

        def counted_func(x):
            if call_count[0] >= max_calls:
                return 1e10  # Return large value to stop optimization
            call_count[0] += 1
            x = np.clip(x, -5, 5)
            try:
                val = float(func(x))
                if not np.isfinite(val):
                    return 1e10
                return val
            except Exception:
                return 1e10

        # Track best solution found
        best_x = np.zeros(self.dim)
        best_f = float('inf')

        # Initial broad sampling to get a good starting point
        rng = np.random.default_rng(42)

        # Phase 1: Sample several diverse starting points
        n_initial = min(10, max(5, max_calls // 50))
        candidates = []
        for _ in range(n_initial):
            x = rng.uniform(-5, 5, self.dim)
            fx = counted_func(x)
            candidates.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort by function value
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Run Powell's method from best starting points
        # Powell's method is derivative-free and works well for Rosenbrock
        n_restarts = min(5, max(2, max_calls // 100))

        for i in range(n_restarts):
            if call_count[0] >= max_calls:
                break
            if time.time() > deadline - 30:
                break

            # Use the i-th best candidate as starting point
            if i < len(candidates):
                x0 = candidates[i][1].copy()
            else:
                x0 = best_x.copy()

            # Run Powell's method with tight tolerances
            try:
                result = minimize(
                    counted_func,
                    x0,
                    method='Powell',
                    options={
                        'maxiter': max(10, (max_calls - call_count[0]) // 20),
                        'maxfev': max(50, max_calls - call_count[0] - 10),
                        'xtol': 1e-10,
                        'ftol': 1e-12
                    }
                )

                if result.success or result.fun < best_f:
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()

            except Exception:
                pass

            # Check time
            if time.time() > deadline - 20:
                break

        # Phase 3: If we have budget remaining, do local refinement with Nelder-Mead
        if call_count[0] < max_calls and time.time() < deadline - 15:
            remaining = max_calls - call_count[0]
            if remaining > 30:
                try:
                    result = minimize(
                        counted_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': min(100, remaining // 2),
                            'maxfev': min(remaining - 5, 200),
                            'xtol': 1e-12,
                            'ftol': 1e-14
                        }
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass

        # Ensure we're within bounds
        best_x = np.clip(best_x, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
