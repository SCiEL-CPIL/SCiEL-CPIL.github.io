"""Model-and-optimize approach for BBOB f20 (Schwefel) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Strategy:
    1. Quick multi-start L-BFGS-B from several random points to get a good local solution fast.
    2. Refine with (1+1)-ES with 1/5 success rule (inspired by simple ES).
    3. Explicit deadline guard to stay within time limits.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._start_time = None
        self._deadline = None
        self._safety_margin = 30  # seconds

    def __call__(self, func):
        self._start_time = time.time()
        self._deadline = self._start_time + 330 - self._safety_margin
        self._call_count = 0
        self.budget_remaining = self.budget

        def safe_call(x):
            """Wrapper to track calls and enforce budget/time limits."""
            nonlocal self
            if self._call_count >= self.budget:
                return 1e10  # Return a large value to discourage further exploration
            if time.time() >= self._deadline:
                return 1e10
            self._call_count += 1
            self.budget_remaining -= 1
            val = func(x)
            if not np.isfinite(val):
                return 1e10
            return val

        dim = self.dim
        rng = np.random.default_rng(42)

        # Track best solution found so far
        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Quick multi-start with L-BFGS-B
        # Try a few different starting points with L-BFGS-B
        num_starts = 5
        for i in range(num_starts):
            if self._call_count >= self.budget or time.time() >= self._deadline:
                break

            # Generate starting point
            if i == 0:
                x0 = np.zeros(dim)
            elif i == 1:
                x0 = rng.uniform(-3, 3, dim)
            elif i == 2:
                x0 = rng.uniform(-4, 4, dim)
            else:
                # Random point
                x0 = rng.uniform(-5, 5, dim)

            # Clip to bounds
            x0 = np.clip(x0, -5, 5)

            # Try L-BFGS-B
            try:
                from scipy.optimize import minimize

                result = minimize(
                    safe_call,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
                )

                if result.success or result.fun < best_f:
                    update_best(result.x, result.fun)
            except Exception:
                pass

            # If we're running low on budget, skip remaining starts
            if self._call_count >= self.budget * 0.6:
                break

        # Ensure we have a best solution
        if best_x is None:
            x0 = rng.uniform(-5, 5, dim)
            f0 = safe_call(x0)
            best_x = x0.copy()
            best_f = f0

        # Phase 2: Refine with (1+1)-ES using 1/5 success rule
        if self._call_count < self.budget and time.time() < self._deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 2.0  # Larger initial step size from inspiration 2

            # 1/5 success rule parameters
            success_count = 0
            total_count = 0
            window_size = 5

            while self._call_count < self.budget and time.time() < self._deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)

                if fy < fx:
                    x = y
                    fx = fy
                    success_count += 1
                    update_best(x, fx)
                else:
                    pass

                total_count += 1

                # Adjust sigma based on 1/5 success rule
                if total_count >= window_size:
                    success_rate = success_count / total_count
                    if success_rate > 0.2:  # More than 1/5 successes
                        sigma *= 1.05
                    elif success_rate < 0.2:  # Less than 1/5 successes
                        sigma *= 0.95
                    success_count = 0
                    total_count = 0

                # Clamp sigma
                sigma = np.clip(sigma, 0.01, 5.0)

        # If we still have budget, try a few more L-BFGS-B from best found
        if self._call_count < self.budget * 0.8 and time.time() < self._deadline:
            try:
                from scipy.optimize import minimize

                result = minimize(
                    safe_call,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )

                if result.fun < best_f:
                    update_best(result.x, result.fun)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
