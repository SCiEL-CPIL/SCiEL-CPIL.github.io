"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with deadline guard for Rastrigin-rotated f15."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 300s limit, 15s safety margin
        dim = self.dim
        budget = self.budget

        # Phase 1: Global exploration with multiple random starts
        # Use ~30% of budget for global search
        global_budget = max(50, budget // 3)
        n_starts = 3
        per_start = global_budget // n_starts

        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)

        for s in range(n_starts):
            if time.time() >= deadline:
                break
            x = rng.uniform(-4, 4, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Brief local refinement from this start
            sigma = 1.5
            for i in range(per_start - 1):
                if time.time() >= deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.75

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        # Phase 2: Intensive local search around the best point
        # Use remaining budget for fine-tuning
        remaining = budget - global_budget
        x = best_x.copy()
        fx = best_f
        sigma = 0.5

        # Adaptive step size based on success rate
        success_count = 0
        window = 10

        for i in range(remaining):
            if time.time() >= deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.15
                success_count += 1
            else:
                sigma *= 0.85

            # Adjust sigma based on recent success rate
            if (i + 1) % window == 0:
                rate = success_count / window
                if rate > 0.4:
                    sigma *= 1.3
                elif rate < 0.1:
                    sigma *= 0.7
                success_count = 0

            # Keep global best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
