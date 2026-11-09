"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with dedicated refinement phase."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin from 300s

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            calls += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multi-start exploration
        # Use ~70% of budget for exploration with 5 different starts
        n_starts = 5
        explore_budget = int(budget * 0.70)
        per_start = explore_budget // n_starts

        for s in range(n_starts):
            if calls >= budget or time.time() > deadline:
                break
            # Different starting points
            if s == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-5, 5, dim)
            
            fx = evaluate(x)
            sigma = 1.5

            for _ in range(per_start):
                if calls >= budget or time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.3, 5.0)
                else:
                    sigma *= 0.85

        # Phase 2: Dedicated exploitation with smaller sigma
        if best_x is not None and calls < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            refine_budget = budget - calls

            for _ in range(refine_budget):
                if calls >= budget or time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.82
                    # If sigma gets too small, try a slightly larger step
                    if sigma < 0.01:
                        sigma = 0.05

        return best_x, best_f
# EVOLVE-BLOCK-END
