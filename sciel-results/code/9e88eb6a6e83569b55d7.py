"""Optimizer for BBOB f23 (Katsuura) with LHS initialization and adaptive (1+1)-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with LHS initialization and deadline guard for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Latin Hypercube sampling for diverse initial points
        # Use a subset of budget for exploration
        n_lhs = min(50, budget // 10)
        lhs_points = []
        for i in range(dim):
            perm = rng.permutation(n_lhs)
            points = (perm + rng.uniform(size=n_lhs)) / n_lhs * 10.0 - 5.0
            lhs_points.append(points)
        lhs_points = np.array(lhs_points).T  # shape (n_lhs, dim)

        best_x = None
        best_f = np.inf
        calls_made = 0

        for i in range(n_lhs):
            if time.time() > deadline:
                break
            x = lhs_points[i]
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            # Fallback: single random point
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            best_x = x
            best_f = fx

        # Phase 2: (1+1)-ES from best point
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        s = 0.0  # success history
        mu = 0.84 * dim + 0.015 * (dim - 2)  # learning rate for sigma
        c_sigma = (mu + 2) / (4 * (dim + 1.3) + mu)
        h_sigma = np.sqrt((1 + c_sigma) / (dim + 1)) / (1 - c_sigma / 2)
        successes = 0
        total_steps = 0

        remaining_budget = budget - calls_made
        for step in range(remaining_budget):
            if time.time() > deadline:
                break

            # Generate step
            direction = rng.standard_normal(dim)
            y = np.clip(x + sigma * direction, -5, 5)
            fy = func(y)
            calls_made += 1
            total_steps += 1

            # Success history based adaptation
            success = fy < fx
            s = (1 - c_sigma) * s + c_sigma * (1.0 if success else 0.0)

            if success:
                x = y
                fx = fy
                successes += 1

            # Update sigma using success history
            sigma *= np.exp(c_sigma * (s / h_sigma - 1))
            sigma = max(sigma, 1e-8)
            sigma = min(sigma, 10.0)

            # Adaptive step size adjustment based on success rate
            # Simple heuristic: if success rate is too high, increase sigma
            # If too low, decrease sigma
            if total_steps > 20:
                success_rate = successes / total_steps
                if success_rate > 0.5:
                    sigma *= 1.1
                elif success_rate < 0.1:
                    sigma *= 0.9

        return x, fx
# EVOLVE-BLOCK-END
