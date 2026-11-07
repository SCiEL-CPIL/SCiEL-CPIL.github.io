"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with per-dimension adaptive step sizes and final polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        num_calls = 1

        # Per-dimension step sizes, initialized to cover the box
        sigmas = np.full(dim, 2.0)
        # Track recent successes per dimension for adaptive scaling
        success_counts = np.zeros(dim)
        total_steps = 0
        window_size = 20

        def check_time():
            return time.time() > deadline

        # Main (1+1)-ES phase
        for _ in range(budget - 1):
            if num_calls >= budget or check_time():
                break

            # Generate perturbation
            perturbation = sigmas * rng.standard_normal(dim)
            y = np.clip(x + perturbation, -5, 5)
            fy = func(y)
            num_calls += 1

            if fy < fx:
                x = y
                fx = fy
                # Increase step sizes where movement was successful
                for d in range(dim):
                    if abs(perturbation[d]) > 1e-10:
                        success_counts[d] += 1
                        sigmas[d] *= 1.15
                total_steps += 1
            else:
                # Decrease step sizes, more aggressively for dimensions
                # that have been failing
                for d in range(dim):
                    sigmas[d] *= 0.85
                total_steps += 1

            # Adapt based on success rate in recent window
            if total_steps > window_size:
                success_rate = total_steps / (2 * window_size)  # rough estimate
                if success_rate > 0.5:
                    # Too many successes, increase exploration
                    sigmas *= 1.05
                elif success_rate < 0.2:
                    # Too many failures, decrease step sizes
                    sigmas *= 0.9

            # Clamp step sizes
            sigmas = np.clip(sigmas, 1e-6, 5.0)

            if fy < best_f:
                best_f = fy
                best_x = y.copy()

        # Final polish phase: use remaining budget with very small steps
        remaining = budget - num_calls
        if remaining > 0 and not check_time():
            polish_sigma = np.full(dim, 0.1)
            x_polish = best_x.copy()
            fx_polish = best_f
            for _ in range(remaining):
                if check_time():
                    break
                perturbation = polish_sigma * rng.standard_normal(dim)
                y = np.clip(x_polish + perturbation, -5, 5)
                fy = func(y)
                num_calls += 1
                if fy < fx_polish:
                    x_polish = y
                    fx_polish = fy
                    polish_sigma *= 1.1
                else:
                    polish_sigma *= 0.8
                polish_sigma = np.clip(polish_sigma, 1e-8, 0.5)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
