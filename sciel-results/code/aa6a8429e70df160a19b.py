"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with adaptive population for Rastrigin rotated."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        # Phase 1: Broad random search to get a decent starting point
        n_init = min(20, budget // 10)
        for i in range(n_init):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            return (np.zeros(dim), 0.0)

        # Phase 2: CMA-ES-like evolution strategy with adaptive step size
        x = best_x.copy()
        fx = best_f
        sigma = 1.0

        # Adaptive population size
        lambda_ = 4 + int(3 * np.log(dim))  # ~8 for dim=5
        mu = lambda_ // 2

        # Step size adaptation (1/5 rule generalized)
        success_rate_target = 1.0 / 5.0
        sigma_history = []

        calls_used = n_init
        while calls_used < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate offspring
            offspring = []
            for j in range(lambda_):
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                offspring.append((fy, y))
                calls_used += 1

            if not offspring:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Update best
            if offspring[0][0] < fx:
                fx = offspring[0][0]
                x = offspring[0][1].copy()
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            # Step size adaptation: increase if many improvements, decrease otherwise
            n_success = sum(1 for fy, _ in offspring if fy < fx)
            success_rate = n_success / len(offspring)
            if success_rate > success_rate_target:
                sigma *= 1.2
            else:
                sigma *= 0.85

            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 0.01, 5.0)

            # Restart if stuck for a while
            if sigma < 0.05 and calls_used < budget * 0.8:
                # Try a new random restart
                x = rng.uniform(-5, 5, dim)
                fx = func(x)
                calls_used += 1
                sigma = 1.0
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return (best_x, best_f)
# EVOLVE-BLOCK-END
