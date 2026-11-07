"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-like optimizer with adaptive step size and coordinate perturbations."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        rng = np.random.default_rng(12345)
        d = self.dim

        # Initial point
        x = np.zeros(d)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        # Adaptive sigma
        sigma = 1.5
        # Success counter for 1/5 rule
        successes = 0
        total_trials = 0

        # Coordinate directions for diversification
        for i in range(self.budget - 1):
            # Check time budget
            if time.time() > deadline:
                break

            # Generate perturbation
            # Mix of full-dimensional noise and coordinate-specific moves
            r = rng.random()
            if r < 0.15:
                # Coordinate move
                idx = rng.integers(0, d)
                delta = np.zeros(d)
                delta[idx] = sigma * rng.standard_normal()
            elif r < 0.30:
                # Pair move
                idx1 = rng.integers(0, d)
                idx2 = rng.integers(0, d)
                if idx1 == idx2:
                    idx2 = (idx1 + 1) % d
                delta = np.zeros(d)
                delta[idx1] = sigma * rng.standard_normal()
                delta[idx2] = sigma * rng.standard_normal()
            else:
                # Full-dimensional move
                delta = sigma * rng.standard_normal(d)

            y = np.clip(x + delta, -5.0, 5.0)
            fy = func(y)
            calls += 1

            if fy < fx:
                # Success
                x = y
                fx = fy
                successes += 1
                # Increase sigma slightly
                sigma *= 1.1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # Failure - try a smaller step in a random direction
                # Occasionally accept a worse point to escape local minima (annealing-like)
                if rng.random() < 0.05 and calls < self.budget * 0.8:
                    # Probabilistic acceptance of worse solutions
                    diff = fx - fy  # positive since fy > fx
                    if diff < 0.5 * sigma:  # only accept if not too far
                        x = y
                        fx = fy
                else:
                    sigma *= 0.9

            total_trials += 1

            # Adapt sigma based on success rate (1/5 rule)
            if total_trials >= 10:
                success_rate = successes / total_trials
                if success_rate > 0.2:
                    sigma *= 1.05
                elif success_rate < 0.1:
                    sigma *= 0.9
                # Reset counters
                successes = 0
                total_trials = 0

            # Clamp sigma
            sigma = np.clip(sigma, 0.01, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
