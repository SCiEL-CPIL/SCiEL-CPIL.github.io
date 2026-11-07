"""Rewritten optimizer using CMA-ES style adaptive covariance approach."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES-inspired optimizer with adaptive step sizes and momentum."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Initial point
        x = rng.uniform(-4, 4, n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Adaptive step size
        sigma = 1.5
        sigma_min = 1e-4
        sigma_max = 5.0

        # Track recent improvements for adaptive sigma
        success_count = 0
        total_count = 0
        window_size = 10

        # Momentum for direction
        velocity = np.zeros(n)
        momentum = 0.7

        calls = 1

        for i in range(budget - 1):
            if calls >= budget:
                break

            # Generate candidate with adaptive sigma
            noise = rng.standard_normal(n)
            # Blend random exploration with momentum direction
            if np.linalg.norm(velocity) > 1e-8:
                dir_unit = velocity / np.linalg.norm(velocity)
                noise = 0.7 * noise + 0.3 * dir_unit * np.linalg.norm(noise)

            candidate = x + sigma * noise
            candidate = np.clip(candidate, -5, 5)

            fy = func(candidate)
            calls += 1

            if fy < fx:
                # Improvement
                velocity = momentum * velocity + (1 - momentum) * (candidate - x)
                x = candidate
                fx = fy
                sigma = min(sigma * 1.2, sigma_max)
                success_count += 1

                if fy < best_fx:
                    best_fx = fy
                    best_x = candidate.copy()
            else:
                # No improvement
                velocity = momentum * velocity
                sigma = max(sigma * 0.9, sigma_min)

            total_count += 1

            # Periodically adjust sigma based on success rate
            if total_count % window_size == 0:
                success_rate = success_count / total_count
                if success_rate > 0.7:
                    sigma = min(sigma * 1.1, sigma_max)
                elif success_rate < 0.3:
                    sigma = max(sigma * 0.9, sigma_min)
                success_count = 0
                total_count = 0

            # Occasionally try a larger jump to escape local optima
            if i % 50 == 49 and sigma < 3.0:
                big_noise = rng.standard_normal(n) * 2.0
                big_candidate = np.clip(x + big_noise, -5, 5)
                if calls < budget:
                    big_fy = func(big_candidate)
                    calls += 1
                    if big_fy < fx:
                        x = big_candidate
                        fx = big_fy
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                        sigma = min(sigma * 1.5, sigma_max)

        return best_x, best_fx
# EVOLVE-BLOCK-END
