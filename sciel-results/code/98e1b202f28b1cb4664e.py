"""Optimizer for BBOB-style f5 (Linear slope) with adaptive CMA-ES-like search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based optimizer with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Time guard
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # Population size
        lam = max(4, int(4 + 3 * np.log(d)))

        # Initial state
        x_best = rng.uniform(-5, 5, d)
        f_best = func(x_best)
        calls = 1

        # Adaptive step size
        sigma = 2.0
        sigma_min = 1e-10
        sigma_max = 10.0

        # Momentum for step size adaptation
        successes = 0
        window_size = 5
        success_count = 0

        # Main loop
        while calls < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            population = []
            for i in range(lam):
                if calls + 1 >= budget:
                    break
                if time.time() > deadline:
                    break
                step = sigma * rng.standard_normal(d)
                candidate = np.clip(x_best + step, -5, 5)
                f_val = func(candidate)
                calls += 1
                population.append((f_val, candidate))

            if not population:
                break

            # Sort by fitness
            population.sort(key=lambda x: x[0])
            best_f, best_x = population[0]

            # Update best
            if best_f < f_best:
                f_best = best_f
                x_best = best_x
                success_count += 1
            else:
                success_count = 0

            # Adaptive step size: increase if many successes, decrease otherwise
            if success_count >= window_size:
                sigma = min(sigma * 1.3, sigma_max)
                success_count = 0
            elif success_count == 0:
                sigma = max(sigma * 0.8, sigma_min)

            # Occasionally restart with larger step to escape local minima
            if calls % (lam * 10) == 0 and sigma < 0.5:
                # Try a larger random step
                if calls + 1 < budget and time.time() < deadline:
                    large_step = rng.uniform(2, 5, d)
                    large_step *= np.sign(rng.standard_normal(d))
                    candidate = np.clip(x_best + large_step, -5, 5)
                    f_val = func(candidate)
                    calls += 1
                    if f_val < f_best:
                        f_best = f_val
                        x_best = candidate
                        sigma = min(sigma * 2, sigma_max)

            # If sigma is very small, try a random restart
            if sigma < sigma_min * 10:
                if calls + 1 < budget and time.time() < deadline:
                    candidate = rng.uniform(-5, 5, d)
                    f_val = func(candidate)
                    calls += 1
                    if f_val < f_best:
                        f_best = f_val
                        x_best = candidate
                        sigma = 1.0
                else:
                    break

        return x_best, f_best
# EVOLVE-BLOCK-END
