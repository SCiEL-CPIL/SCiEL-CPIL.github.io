"""Optimized optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with aggressive early exploration and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Broad exploration with random samples to find a good basin
        # Use ~20% of budget for initial exploration
        explore_budget = max(20, budget // 5)
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        calls = 1

        x_best = x.copy()
        fx_best = fx

        # Sample several random points to find a better starting point
        for i in range(explore_budget - 1):
            if time.time() > deadline:
                break
            y = rng.uniform(-5, 5, dim)
            fy = func(y)
            calls += 1
            if fy < fx_best:
                x_best = y.copy()
                fx_best = fy

        # Phase 2: Focused (1+1)-ES from the best point found
        x = x_best.copy()
        fx = fx_best
        sigma = 1.5  # moderate initial step size
        sigma_min = 1e-6
        sigma_max = 5.0

        # 1/5 success rule parameters
        c_sigma = 1.0 / 5.0  # adaptation rate
        d_sigma = 1.0 + 0.35 + 2.0 / (dim + 1.7)  # damping

        # Track consecutive successes for step size adaptation
        successes = 0
        total_trials = 0

        remaining = budget - calls

        for i in range(remaining):
            if time.time() > deadline:
                break

            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                # Success: accept and increase step size
                x, fx = y, fy
                successes += 1
                sigma = min(sigma * 1.2, sigma_max)

                if fy < fx_best:
                    x_best, fx_best = y.copy(), fy
            else:
                # Failure: decrease step size
                sigma = max(sigma * 0.8, sigma_min)

            total_trials += 1

            # Periodically adapt sigma using 1/5 rule
            if total_trials >= 5:
                success_rate = successes / total_trials
                if success_rate > 0.2:
                    sigma *= 1.1
                elif success_rate < 0.15:
                    sigma *= 0.9
                sigma = np.clip(sigma, sigma_min, sigma_max)
                successes = 0
                total_trials = 0

            # Stagnation detection: if no improvement for 50 iterations, restart
            if i > 0 and i % 50 == 0:
                # Check if we've made progress recently
                # Simple heuristic: if sigma is very small, we're stuck
                if sigma < 0.01:
                    # Restart from a new random point, but keep best
                    new_x = rng.uniform(-5, 5, dim)
                    new_fx = func(new_x)
                    calls += 1
                    if new_fx < fx:
                        x, fx = new_x, new_fx
                        sigma = 1.0
                    else:
                        # Try a point near the best found
                        x = x_best + 0.5 * rng.standard_normal(dim)
                        x = np.clip(x, -5, 5)
                        fx = func(x)
                        calls += 1
                        sigma = 0.5

        return x_best, fx_best
# EVOLVE-BLOCK-END
