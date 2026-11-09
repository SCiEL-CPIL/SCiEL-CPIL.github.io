"""Optimizer for BBOB f20 (Schwefel) with hybrid exploration and (1+1)-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: Sobol exploration + adaptive (1+1)-ES with restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # Safety margin before 300s kill

        # Sobol sequence for structured exploration
        from scipy.stats import qmc

        def sobol_points(n, dim):
            sampler = qmc.Sobol(d=dim, scramble=True)
            samples = sampler.random(n)
            return samples * 10.0 - 5.0  # Map to [-5, 5]^dim

        # Phase 1: Structured exploration with Sobol
        n_sobol = min(50, budget // 5)
        best_x = None
        best_f = np.inf

        if n_sobol > 0:
            sobol_xs = sobol_points(n_sobol, dim)
            for i in range(n_sobol):
                if time.time() > deadline:
                    break
                x = sobol_xs[i]
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Also try origin and a few standard points
        standard_points = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            -np.ones(dim) * 2.0,
        ]
        for sp in standard_points:
            if time.time() > deadline:
                break
            fx = func(sp)
            if fx < best_f:
                best_f = fx
                best_x = sp.copy()

        if best_x is None:
            # Fallback: random start
            rng = np.random.default_rng(42)
            best_x = rng.uniform(-5, 5, dim)
            best_f = func(best_x)

        # Phase 2: Adaptive (1+1)-ES with sigma adaptation
        calls_made = n_sobol + len(standard_points)
        remaining = budget - calls_made

        if remaining > 0 and best_x is not None:
            rng = np.random.default_rng(123)
            x = best_x.copy()
            fx = best_f
            sigma = 1.5  # Initial step size
            sigma_min = 1e-6
            sigma_max = 5.0
            success_count = 0
            window_size = 10

            for i in range(remaining):
                if time.time() > deadline:
                    break

                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)

                if fy < fx:
                    x = y
                    fx = fy
                    success_count += 1
                    # Increase step size on success
                    sigma = min(sigma * 1.2, sigma_max)
                else:
                    # Decrease step size on failure
                    sigma = max(sigma * 0.85, sigma_min)

                # Periodic restart if stuck
                if i > 0 and i % window_size == 0:
                    if success_count == 0:
                        # No progress in last window: try a small perturbation
                        x = np.clip(x + 0.5 * sigma * rng.standard_normal(dim), -5, 5)
                        fx = func(x)
                        sigma = max(sigma * 0.5, sigma_min)
                    success_count = 0

                # Update best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
