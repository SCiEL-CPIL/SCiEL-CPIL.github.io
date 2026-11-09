"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES-like optimizer with adaptive step size and refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin from 300s

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            calls += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_ok():
            return time.time() < deadline

        # Phase 1: Multi-start exploration with (1+1)-ES
        # Use ~60% of budget for exploration
        n_starts = 6
        explore_budget = int(budget * 0.60)
        per_start = max(1, explore_budget // n_starts)

        # Different starting points for diversity
        starts = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            -np.ones(dim) * 2.0,
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-3, 3, dim),
        ]

        for s in range(n_starts):
            if calls >= budget or not time_ok():
                break

            x = starts[s].copy()
            fx = evaluate(x)
            if not time_ok() or calls >= budget:
                break

            sigma = 1.0
            success_count = 0
            total_trials = 0
            success_threshold = 0.2  # adapt sigma based on success rate

            for _ in range(per_start):
                if calls >= budget or not time_ok():
                    break

                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                total_trials += 1

                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                else:
                    success_count = max(0, success_count - 1)

                # Adapt sigma based on recent success rate
                if total_trials >= 5:
                    rate = success_count / total_trials
                    if rate > 0.35:
                        sigma *= 1.1
                    elif rate < 0.15:
                        sigma *= 0.85
                    else:
                        sigma *= 0.98  # slight contraction

                # Keep sigma in reasonable bounds
                sigma = np.clip(sigma, 0.01, 5.0)

        # Phase 2: Focused refinement around the best point
        if best_x is not None and calls < budget and time_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.15
            refine_budget = budget - calls

            # Use a more aggressive adaptation for refinement
            for _ in range(refine_budget):
                if calls >= budget or not time_ok():
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                else:
                    sigma *= 0.9

                # Keep sigma in reasonable bounds for refinement
                sigma = np.clip(sigma, 0.001, 1.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
