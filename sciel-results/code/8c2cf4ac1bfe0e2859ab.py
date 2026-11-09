"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with large initial sigmas for Rastrigin escape."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        n_calls = 0
        dim = self.dim
        budget = self.budget

        # Track best solution globally
        best_x = None
        best_f = np.inf

        def track(x, f):
            nonlocal best_x, best_f, n_calls
            n_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Multi-start strategy: multiple random starts with different sigmas
        # For Rastrigin, large initial sigma helps escape local minima
        rng = np.random.default_rng(42)

        # Phase 1: Broad exploration with multiple restarts
        # Use large sigmas to jump between basins
        num_starts = 5
        calls_per_start = budget // num_starts  # 100 calls per start
        remaining = budget

        for start_idx in range(num_starts):
            if n_calls >= budget:
                break
            if time.time() > deadline - 10:
                break

            # Random start in the box
            x = rng.uniform(-4, 4, dim)
            fx = track(x, func(np.clip(x, -5, 5)))

            # CMA-ES-like adaptation for this start
            sigma = 1.5  # Large initial sigma for Rastrigin
            C = np.eye(dim)
            step_size = sigma
            success_count = 0
            window = 10
            calls_in_start = 0
            max_calls_this_start = min(calls_per_start, budget - n_calls)

            for i in range(max_calls_this_start):
                if n_calls >= budget:
                    break
                if time.time() > deadline - 5:
                    break

                # Generate trial point
                y = x + step_size * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = track(y, func(y))

                if fy < fx:
                    # Success: move to new point
                    x = y
                    fx = fy
                    step_size *= 1.2
                    success_count += 1
                else:
                    step_size *= 0.9

                # Adapt step size based on success rate over window
                if (i + 1) % window == 0:
                    success_rate = success_count / window
                    if success_rate > 0.5:
                        step_size *= 1.1
                    elif success_rate < 0.2:
                        step_size *= 0.8
                    success_count = 0

                calls_in_start += 1

            # Refinement phase for this start: smaller steps
            if n_calls < budget and time.time() < deadline - 5:
                refine_budget = min(20, budget - n_calls)
                step_size = 0.1
                for i in range(refine_budget):
                    if n_calls >= budget:
                        break
                    if time.time() > deadline - 5:
                        break
                    y = x + step_size * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    fy = track(y, func(y))
                    if fy < fx:
                        x = y
                        fx = fy

        # Final refinement around the global best
        if best_x is not None and n_calls < budget and time.time() < deadline - 5:
            final_budget = min(30, budget - n_calls)
            step_size = 0.05
            for i in range(final_budget):
                if n_calls >= budget:
                    break
                if time.time() > deadline - 5:
                    break
                y = best_x + step_size * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = track(y, func(y))
                if fy < best_f:
                    best_f = fy
                    best_x = y
                    step_size *= 0.8
                else:
                    step_size *= 0.95

        return best_x, best_f
# EVOLVE-BLOCK-END
