"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, restarts, and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s limit

        rng = np.random.default_rng(0)

        # Global best
        x_best = rng.uniform(-5, 5, self.dim)
        fx_best = func(x_best)

        calls_made = 1
        remaining = self.budget - 1

        # Parameters for (1+1)-ES with restarts
        sigma = 1.0
        sigma_min = 1e-4
        sigma_max = 10.0
        stall_limit = 50  # Restart after this many unsuccessful steps
        stall_count = 0

        # Track best for final return
        x_nm_best = x_best.copy()
        fx_nm_best = fx_best

        # Phase 1: (1+1)-ES with restarts for global exploration
        # Use about 60% of budget for exploration
        explore_budget = int(0.6 * self.budget)
        explore_calls = 0

        while (remaining > 0 and time.time() < deadline and explore_calls < explore_budget):
            y = np.clip(x_best + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_made += 1
            remaining -= 1
            explore_calls += 1

            if fy < fx_best:
                x_best, fx_best = y, fy
                sigma = min(sigma * 1.2, sigma_max)
                stall_count = 0
                if fx_best < fx_nm_best:
                    x_nm_best, fx_nm_best = x_best.copy(), fx_best
            else:
                sigma = max(sigma * 0.9, sigma_min)
                stall_count += 1

                if stall_count >= stall_limit:
                    # Restart from a new random point
                    stall_count = 0
                    sigma = 1.0
                    x_new = rng.uniform(-5, 5, self.dim)
                    fx_new = func(x_new)
                    calls_made += 1
                    remaining -= 1
                    explore_calls += 1
                    if fx_new < fx_best:
                        x_best, fx_best = x_new, fx_new
                        if fx_best < fx_nm_best:
                            x_nm_best, fx_nm_best = x_best.copy(), fx_best
                    else:
                        # Keep current best, just reset sigma
                        pass

        # Phase 2: Use remaining budget for local refinement around the best point
        if remaining > 0 and time.time() < deadline:
            # Local search with decreasing step size
            local_sigma = max(sigma * 0.5, 0.1)
            local_calls = 0
            local_budget = remaining

            while local_calls < local_budget and time.time() < deadline:
                # Decrease step size over time
                progress = local_calls / local_budget
                current_sigma = local_sigma * (1.0 - 0.9 * progress)
                current_sigma = max(current_sigma, 1e-4)

                y = np.clip(x_nm_best + current_sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1
                local_calls += 1

                if fy < fx_nm_best:
                    x_nm_best, fx_nm_best = y, fy

        return x_nm_best, fx_nm_best
# EVOLVE-BLOCK-END
