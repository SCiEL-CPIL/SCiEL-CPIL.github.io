"""Optimized optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with 1/5 success rule and adaptive restarts."""

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
        best_x = x.copy()
        best_f = fx

        # Track successes for 1/5 rule
        successes = 0
        trials = 0

        while calls < max_calls:
            if time.time() > deadline - 1.0:
                break

            # Generate a trial point
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1
            trials += 1

            if fy < fx:
                # Success: move to y
                x = y
                fx = fy
                successes += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            # Apply 1/5 rule every 5 trials
            if trials >= 5:
                if successes >= 2:
                    sigma *= 1.5  # more successes -> increase step
                elif successes == 0:
                    sigma *= 0.75  # no successes -> decrease step
                successes = 0
                trials = 0

            # Keep sigma in reasonable bounds
            sigma = max(1e-10, min(sigma, 10.0))

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        dim = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        total_calls = 0
        budget = self.budget

        # Phase 1: Multi-start exploration
        # Use several random starts with the (1+1)-ES
        n_starts = 5
        calls_per_start = budget // n_starts

        for i in range(n_starts):
            if total_calls >= budget or time.time() > deadline - 30.0:
                break

            # Random start points
            if i == 0:
                x0 = rng.uniform(-2, 2, dim)
            elif i == 1:
                x0 = rng.uniform(-4, 4, dim)
            elif i == 2:
                x0 = rng.uniform(-5, 5, dim)
            elif i == 3:
                # Start near the best found so far if we have one
                if best_x is not None:
                    x0 = best_x + 1.0 * rng.standard_normal(dim)
                    x0 = np.clip(x0, -5, 5)
                else:
                    x0 = rng.uniform(-5, 5, dim)
            else:
                x0 = rng.uniform(-5, 5, dim)

            sigma0 = 2.0 if i < 2 else 3.0

            remaining = min(calls_per_start, budget - total_calls)
            if remaining < 5:
                break

            x, fx, calls_used = self._es_1plus1(
                func, rng, x0, sigma0, remaining, deadline
            )
            total_calls += calls_used

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Local refinement from the best point with smaller sigma
        if best_x is not None and total_calls < budget:
            remaining = budget - total_calls
            if remaining > 10:
                # Use a smaller sigma for fine-tuning
                x, fx, calls_used = self._es_1plus1(
                    func, rng, best_x, 0.3, remaining, deadline
                )
                total_calls += calls_used
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Phase 3: If we still have budget, do another round of refinement
        if best_x is not None and total_calls < budget:
            remaining = budget - total_calls
            if remaining > 10:
                x, fx, calls_used = self._es_1plus1(
                    func, rng, best_x, 0.1, remaining, deadline
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
