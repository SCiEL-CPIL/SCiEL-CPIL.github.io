"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with initial exploration, adaptive sigma, and restart logic."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        deadline = time.time() + 290.0  # safety margin before 300s kill

        # Track best
        best_x = None
        best_f = np.inf
        calls = 0

        def safe_call(x):
            nonlocal calls, best_x, best_f
            x = np.asarray(x, dtype=float).reshape(dim)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Stage 1: Brief coarse exploration (10 points) to find a good starting basin
        n_coarse = min(10, budget)
        for _ in range(n_coarse):
            if time.time() > deadline or calls >= budget:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            safe_call(x)

        if calls >= budget:
            return best_x, best_f

        # Stage 2: (1+1)-ES with adaptive sigma and restart logic
        x = best_x.copy()
        fx = best_f
        sigma = 1.5
        success_count = 0
        no_improve = 0
        max_no_improve = 40
        restarts_done = 0
        max_restarts = 6

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
            fy = safe_call(y)

            if fy < fx:
                x = y.copy()
                fx = fy
                success_count += 1
                no_improve = 0
            else:
                no_improve += 1

            # 1/5 success rule: adjust sigma every dim+1 steps
            if success_count >= dim + 1:
                rate = success_count / (dim + 1)
                if rate > 0.2:
                    sigma *= 1.25
                elif rate < 0.2:
                    sigma *= 1.0 / 1.25
                sigma = np.clip(sigma, 1e-5, 4.0)
                success_count = 0

            # Restart logic: if stuck, try a new region
            if no_improve >= max_no_improve:
                no_improve = 0
                restarts_done += 1
                if restarts_done <= max_restarts:
                    # Mix of local jump and occasional global restart
                    if rng.random() < 0.25:
                        x = rng.uniform(-5.0, 5.0, dim)
                    else:
                        x = np.clip(x + 1.5 * rng.standard_normal(dim), -5.0, 5.0)
                    fx = safe_call(x)
                    sigma = max(sigma, 0.8)
                    success_count = 0
                else:
                    # Just shrink sigma further
                    sigma *= 0.7

            # Gradually shrink sigma in later stages for refinement
            frac_used = calls / budget
            if frac_used > 0.75:
                sigma *= 0.995

        # Final polish: very small steps around best
        if calls < budget and best_x is not None:
            n_polish = min(5, budget - calls)
            for _ in range(n_polish):
                if time.time() > deadline or calls >= budget:
                    break
                step = 0.02 * rng.standard_normal(dim)
                y = np.clip(best_x + step, -5.0, 5.0)
                safe_call(y)

        return best_x, best_f
# EVOLVE-BLOCK-END
