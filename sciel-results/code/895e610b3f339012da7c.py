"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+λ)-ES with adaptive step size and time guard for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit

        rng = np.random.default_rng(42)
        n = self.dim

        # Initial point
        x = rng.uniform(-5, 5, n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Parameters for (1+λ)-ES
        lam = 8
        mu = 4

        # Adaptive sigma
        sigma = 1.0
        sigma_min = 1e-8
        sigma_max = 10.0

        # Compute weights for lambda/2 selection
        # Using log-rank weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = (weights.sum() ** 2) / np.sum(weights ** 2)

        calls_made = 1

        for generation in range(self.budget - 1):
            # Check time limit
            if time.time() > deadline:
                break

            # Generate offspring
            offspring = []
            for i in range(lam):
                if time.time() > deadline:
                    break
                step = sigma * rng.standard_normal(n)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                calls_made += 1
                offspring.append((fy, y))

            if not offspring:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Select best mu
            selected = offspring[:min(mu, len(offspring))]

            # Update mean
            new_x = np.zeros(n)
            for w, (f, y) in zip(weights, selected):
                new_x += w * y

            # Check if we improved
            new_fx = selected[0][0]
            if new_fx < best_fx:
                best_fx = new_fx
                best_x = selected[0][1].copy()

            # Update sigma based on success
            improved = new_fx < fx
            if improved:
                # Move toward the new best
                x = new_x
                fx = new_fx
                # Increase sigma slightly
                sigma = min(sigma * 1.1, sigma_max)
            else:
                # Move toward new_x but keep current best
                # Blend: move half way
                x = 0.5 * x + 0.5 * new_x
                # Decrease sigma
                sigma = max(sigma * 0.9, sigma_min)

        return best_x, best_fx
# EVOLVE-BLOCK-END
