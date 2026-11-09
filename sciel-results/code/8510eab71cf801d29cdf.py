"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with LHS exploration, adaptive sigma, and polish phase."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: LHS exploration to find a good starting point
        # Use Latin Hypercube Sampling for diverse initial points
        lhs_budget = min(80, budget // 6)
        best_x = None
        best_f = float('inf')

        # Generate LHS points
        n_lhs = lhs_budget
        idx = np.arange(n_lhs)
        # For each dimension, create a permutation and scale to [-5, 5]
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = rng.permutation(n_lhs)
            # Map to [-5, 5] with uniform spacing
            lhs_points[:, d] = -5 + 10.0 * (perm + rng.uniform(size=n_lhs)) / n_lhs

        # Evaluate LHS points
        for i in range(n_lhs):
            if time.time() > deadline:
                break
            x = lhs_points[i]
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Also try a few random points
        for i in range(min(10, budget - n_lhs)):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        # Phase 2: Adaptive (1+1)-ES from the best point
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        calls_made = n_lhs + 10
        success_count = 0
        fail_count = 0

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x, fx = y, fy
                success_count += 1
                fail_count = 0
                # Increase sigma on success
                sigma *= 1.3
            else:
                fail_count += 1
                # Decrease sigma on failure
                sigma *= 0.85

            # If stuck for a while, try a larger jump
            if fail_count >= 20:
                sigma *= 2.0
                fail_count = 0

            # Keep sigma within reasonable bounds
            sigma = np.clip(sigma, 0.001, 5.0)

        # Phase 3: Polish phase - fine-tune with smaller steps
        polish_budget = min(50, budget - calls_made)
        if polish_budget > 0:
            sigma_polish = 0.1
            for _ in range(polish_budget):
                if time.time() > deadline:
                    break
                y = x + sigma_polish * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma_polish *= 1.2
                else:
                    sigma_polish *= 0.8

        return x, fx
# EVOLVE-BLOCK-END
