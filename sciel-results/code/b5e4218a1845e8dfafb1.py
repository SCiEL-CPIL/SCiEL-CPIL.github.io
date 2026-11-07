"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive step size for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        best_x = None
        best_f = float('inf')
        calls_made = 0
        deadline = time.time() + 285  # Safety margin before 300s limit

        def run_local_search(x0, fx0, max_calls):
            nonlocal calls_made, best_x, best_f
            x = x0.copy()
            fx = fx0
            sigma = 1.0
            no_improve = 0
            for _ in range(max_calls):
                if time.time() >= deadline:
                    break
                if calls_made >= self.budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    no_improve = 0
                    sigma *= 1.2
                else:
                    no_improve += 1
                    sigma *= 0.9
                    if no_improve > 20:
                        break
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            return x, fx

        # Phase 1: Broad exploration with multiple restarts
        n_restarts = 5
        calls_per_restart = self.budget // n_restarts - 1

        for i in range(n_restarts):
            if calls_made >= self.budget or time.time() >= deadline:
                break
            x0 = rng.uniform(-5, 5, self.dim)
            fx0 = func(x0)
            calls_made += 1
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()
            remaining = min(calls_per_restart, self.budget - calls_made)
            if remaining > 5:
                run_local_search(x0, fx0, remaining)

        # Phase 2: Refine around best point found
        if best_x is not None and calls_made < self.budget and time.time() < deadline:
            remaining = self.budget - calls_made
            if remaining > 10:
                run_local_search(best_x, best_f, remaining)

        if best_x is None:
            # Fallback: use center
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
