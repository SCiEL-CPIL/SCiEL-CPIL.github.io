"""Optimized optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with 1/5 success rule for ill-conditioned problems."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _es_1plus1(self, func, rng, x0, sigma0, max_calls, deadline):
        """
        (1+1)-ES with 1/5 success rule.
        Returns best_x, best_f, calls_used.
        """
        dim = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        sigma = sigma0
        calls = 1

        while calls < max_calls:
            if time.time() > deadline - 1.0:
                break

            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5  # 1/5 rule: increase on success
            else:
                sigma *= 0.87  # 1/5 rule: decrease on failure

            # Keep sigma in reasonable bounds
            sigma = max(1e-10, min(sigma, 20.0))

        return x, fx, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        dim = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        total_calls = 0
        budget = self.budget

        # Phase 1: Multi-start exploration
        # Use 3-5 starts with sufficient budget for each to converge
        n_starts = min(5, max(2, budget // 100))
        calls_per_start = max(50, (budget - 50) // n_starts)

        for i in range(n_starts):
            if total_calls >= budget or time.time() > deadline - 30.0:
                break

            # Diverse starting points
            if i == 0:
                x0 = rng.uniform(-1, 1, dim)
                sigma0 = 1.5
            elif i == 1:
                x0 = rng.uniform(-3, 3, dim)
                sigma0 = 2.0
            elif i == 2:
                x0 = rng.uniform(-4, 4, dim)
                sigma0 = 2.5
            else:
                x0 = rng.uniform(-5, 5, dim)
                sigma0 = 3.0

            max_calls_this = min(calls_per_start, budget - total_calls)
            if max_calls_this < 10:
                break

            x, fx, calls_used = self._es_1plus1(
                func, rng, x0, sigma0,
                max_calls_this, deadline
            )
            total_calls += calls_used

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Local refinement from best point found
        if best_x is not None and total_calls < budget:
            remaining = budget - total_calls
            if remaining > 20:
                # Refine with smaller initial sigma
                x, fx, calls_used = self._es_1plus1(
                    func, rng, best_x, 0.3,
                    remaining, deadline
                )
                total_calls += calls_used
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Ensure at least one call was made
        if total_calls == 0:
            x = rng.uniform(-5, 5, dim)
            best_f = func(x)
            best_x = x
            total_calls = 1

        return best_x, best_f
# EVOLVE-BLOCK-END
