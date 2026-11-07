"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size and multi-restart strategy."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._calls_made = 0
        self._start_time = 0.0

    def _evaluate(self, func, x):
        """Evaluate function with call counting and deadline guard."""
        if self._calls_made >= self.budget:
            return None
        if time.time() - self._start_time > 320:
            return None
        self._calls_made += 1
        return func(x)

    def _es_search(self, func, x0, sigma, max_calls, rng):
        """(1+1)-ES with 1/5 success rule."""
        x = np.array(x0, dtype=float)
        fx = self._evaluate(func, x)
        if fx is None:
            return x, np.inf

        for _ in range(max_calls - 1):
            if self._calls_made >= self.budget:
                break
            if time.time() - self._start_time > 320:
                break

            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = self._evaluate(func, y)
            if fy is None:
                break

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3
            else:
                sigma *= 0.85

            sigma = max(sigma, 1e-8)
            sigma = min(sigma, 10.0)

        return x, fx

    def __call__(self, func):
        self._start_time = time.time()
        self._calls_made = 0
        rng = np.random.default_rng(12345)

        best_x = None
        best_fx = np.inf

        # Multi-restart strategy with different initializations
        # Phase 1: 3 restarts with broad exploration
        n_restarts = 3
        budget_per_restart = self.budget // n_restarts

        for r in range(n_restarts):
            if self._calls_made >= self.budget:
                break
            if time.time() - self._start_time > 320:
                break

            # Different initialization for each restart
            if r == 0:
                x0 = rng.uniform(-5, 5, self.dim)
            elif r == 1:
                x0 = rng.uniform(-4, 4, self.dim)
            else:
                x0 = np.zeros(self.dim)

            x, fx = self._es_search(func, x0, 2.0, budget_per_restart, rng)
            if fx < best_fx:
                best_fx = fx
                best_x = x

        # Phase 2: Refine the best point with smaller step size
        remaining = self.budget - self._calls_made
        if remaining > 20 and best_x is not None:
            x, fx = self._es_search(func, best_x, 0.5, remaining, rng)
            if fx < best_fx:
                best_fx = fx
                best_x = x

        # Phase 3: Final fine-tuning if budget remains
        remaining = self.budget - self._calls_made
        if remaining > 10 and best_x is not None:
            x, fx = self._es_search(func, best_x, 0.1, remaining, rng)
            if fx < best_fx:
                best_fx = fx
                best_x = x

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_fx = func(best_x)

        return best_x, best_fx
# EVOLVE-BLOCK-END
