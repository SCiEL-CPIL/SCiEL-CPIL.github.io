"""Optimized BBOB f14 solver with adaptive CMA-ES-like strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive optimization for BBOB f14 (Different powers) on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Phase 1: Broad exploration with multiple random starts
        # Evaluate several initial points to get a sense of the landscape
        n_initial = min(10, budget)
        best_x = None
        best_f = np.inf

        for i in range(n_initial):
            if time.time() > deadline:
                break
            if i == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        calls_made = n_initial

        # Phase 2: Adaptive (1+1)-ES with CMA-ES-like covariance adaptation
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        c = np.ones(dim)  # step sizes per dimension
        # Simple adaptive step: track progress

        # Use a more sophisticated approach: adaptive step sizes per dimension
        # with occasional restarts if stuck

        stagnation_count = 0
        max_stagnation = 30

        while calls_made < budget and time.time() < deadline:
            # Generate candidate
            direction = rng.standard_normal(dim)
            y = x + sigma * c * direction
            y = np.clip(y, -5, 5)

            fy = func(y)
            calls_made += 1

            if fy < fx:
                # Accept
                x = y
                fx = fy
                stagnation_count = 0
                # Increase sigma slightly on success
                sigma *= 1.1
                # Adapt c: increase dimensions that helped
                for d in range(dim):
                    if abs(direction[d]) > 0.5:
                        c[d] = min(c[d] * 1.05, 5.0)
            else:
                # Reject
                stagnation_count += 1
                sigma *= 0.9
                # Adapt c: decrease dimensions that didn't help
                for d in range(dim):
                    if abs(direction[d]) > 0.5:
                        c[d] = max(c[d] * 0.95, 0.1)

            # If stuck for too long, do a local restart
            if stagnation_count >= max_stagnation:
                stagnation_count = 0
                max_stagnation = max(20, max_stagnation - 5)
                # Small perturbation restart
                x = x + rng.standard_normal(dim) * 0.5
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_made += 1
                if calls_made >= budget:
                    break

            # Adaptive sigma floor
            sigma = max(sigma, 0.01)

        return x, fx
# EVOLVE-BLOCK-END
