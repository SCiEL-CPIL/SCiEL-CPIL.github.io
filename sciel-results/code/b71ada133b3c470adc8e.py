"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio optimizer: run multiple short (1+1)-ES restarts with different
    strategies, keeping the best found. Uses a time deadline to stay within limits."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _run_1p1_es(self, func, rng, x0, sigma0, n_calls, deadline):
        """Run a (1+1)-ES for n_calls function evaluations or until deadline."""
        x = x0.copy()
        fx = func(x)
        calls_used = 1
        sigma = sigma0

        while calls_used < n_calls:
            if time.time() >= deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_used += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3
            else:
                sigma *= 0.85
            sigma = min(sigma, 10.0)
            sigma = max(sigma, 1e-6)
        return x, fx, calls_used

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # 300s limit, 20s safety margin

        best_x = None
        best_fx = np.inf

        # First, do a quick random search to get a decent starting point
        rng = np.random.default_rng(42)

        # Try several random starts with short (1+1)-ES
        n_restarts = 5
        calls_per_restart = max(1, (budget - n_restarts) // n_restarts)

        for r in range(n_restarts):
            if time.time() >= deadline:
                break
            # Different starting points
            if r == 0:
                x0 = np.zeros(dim)
            elif r == 1:
                x0 = np.full(dim, 2.0)
            elif r == 2:
                x0 = np.full(dim, -2.0)
            elif r == 3:
                x0 = rng.uniform(-4, 4, dim)
            else:
                x0 = rng.uniform(-4, 4, dim)

            sigma0 = 3.0 if r < 3 else 2.0

            x, fx, used = self._run_1p1_es(func, rng, x0, sigma0, calls_per_restart, deadline)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            if best_x is None:
                best_x = x.copy()
                best_fx = fx

        # Use remaining budget for a final polish from the best point
        remaining = budget - sum(
            min(calls_per_restart, budget) for _ in range(n_restarts)
        )
        # Actually, let's just track total calls properly
        # For simplicity, do a final (1+1)-ES with remaining budget
        if best_x is not None and time.time() < deadline:
            # Estimate remaining calls
            # We've used approximately n_restarts * calls_per_restart calls
            used_so_far = min(budget, n_restarts * calls_per_restart)
            remaining_calls = max(1, budget - used_so_far)
            if remaining_calls > 0:
                x, fx, _ = self._run_1p1_es(func, rng, best_x, 1.0, remaining_calls, deadline)
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
