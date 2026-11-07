"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase optimizer: coarse exploration, (1+1)-ES, and local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        # Phase 1: Coarse exploration with multiple random restarts
        # Use ~30% of budget for exploration
        n_explore = max(5, budget // 4)
        best_x = None
        best_f = np.inf
        calls = 0

        for _ in range(n_explore):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: (1+1)-ES from best point found
        x = best_x
        fx = best_f
        sigma = 1.5
        calls_remaining = budget - calls

        for _ in range(calls_remaining):
            if not time_ok():
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.15
            else:
                sigma *= 0.9

            if sigma < 1e-10:
                break

        # Phase 3: Local refinement with very small steps
        sigma = max(sigma, 0.01)
        n_refine = max(10, budget // 10)
        for _ in range(n_refine):
            if not time_ok() or calls >= budget:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= 0.95
            else:
                sigma *= 0.9

            if sigma < 1e-12:
                break

        return x, fx
# EVOLVE-BLOCK-END
