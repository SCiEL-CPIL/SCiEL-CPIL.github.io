"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, periodic exploration, and multi-scale polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        n = self.dim
        budget = self.budget

        # Multi-point initialization for better early exploration
        n_init = min(5, budget)
        candidates = []
        for i in range(n_init):
            x_init = rng.uniform(-5, 5, n)
            fx_init = func(x_init)
            candidates.append((x_init, fx_init))

        # Select best initial point
        best_idx = np.argmin([c[1] for c in candidates])
        x_best = candidates[best_idx][0].copy()
        fx_best = candidates[best_idx][1]
        calls_made = n_init
        remaining = budget - n_init

        # Parameters
        sigma = 1.5
        sigma_min = 1e-5
        sigma_max = 8.0
        stall_limit = 35
        stall_count = 0

        # Track best
        x_nm_best = x_best.copy()
        fx_nm_best = fx_best

        # Periodic exploration: every 30 calls, jump to a new random point
        explore_interval = 30
        calls_since_explore = 0

        # Phase 1: Main (1+1)-ES with periodic exploration
        # Use about 75% of budget for exploration
        explore_budget = int(0.75 * budget)
        explore_calls = 0

        while remaining > 0 and time.time() < deadline and explore_calls < explore_budget:
            # Periodic exploration: jump to a new random point
            calls_since_explore += 1
            if calls_since_explore >= explore_interval:
                calls_since_explore = 0
                # Mix of global and local exploration
                if rng.random() < 0.5:
                    # Global exploration
                    x_jump = rng.uniform(-5, 5, n)
                else:
                    # Local exploration with moderate step
                    x_jump = np.clip(x_best + 2.5 * rng.standard_normal(n), -5, 5)
                fx_jump = func(x_jump)
                calls_made += 1
                remaining -= 1
                explore_calls += 1
                if fx_jump < fx_best:
                    x_best, fx_best = x_jump, fx_jump
                    if fx_best < fx_nm_best:
                        x_nm_best, fx_nm_best = x_best.copy(), fx_best
                    sigma = min(sigma * 1.3, sigma_max)
                    stall_count = 0
                else:
                    pass
                continue

            # Generate candidate
            y = np.clip(x_best + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            calls_made += 1
            remaining -= 1
            explore_calls += 1

            if fy < fx_best:
                x_best, fx_best = y, fy
                if fx_best < fx_nm_best:
                    x_nm_best, fx_nm_best = x_best.copy(), fx_best
                sigma = min(sigma * 1.15, sigma_max)
                stall_count = 0
            else:
                sigma = max(sigma * 0.92, sigma_min)
                stall_count += 1

                if stall_count >= stall_limit:
                    # Restart from new random point
                    stall_count = 0
                    sigma = 1.5
                    x_new = rng.uniform(-5, 5, n)
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

        # Phase 2: Local refinement with decreasing step size
        if remaining > 0 and time.time() < deadline:
            local_sigma = max(sigma * 0.5, 0.1)
            local_calls = 0
            local_budget = remaining

            while local_calls < local_budget and time.time() < deadline:
                # Decrease step size over time
                progress = local_calls / local_budget
                current_sigma = local_sigma * (1.0 - 0.95 * progress)
                current_sigma = max(current_sigma, 1e-6)

                y = np.clip(x_nm_best + current_sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                calls_made += 1
                local_calls += 1

                if fy < fx_nm_best:
                    x_nm_best, fx_nm_best = y, fy

        return x_nm_best, fx_nm_best
# EVOLVE-BLOCK-END
