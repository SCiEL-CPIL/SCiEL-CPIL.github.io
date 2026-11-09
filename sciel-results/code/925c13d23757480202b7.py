"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with adaptive step size for BBOB Sphere."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)

        # Start at origin (center of box)
        x = np.zeros(dim)
        fx = func(x)

        # Adaptive step size
        sigma = 2.5
        # Population size
        lam = 10
        # Track best
        best_x = x.copy()
        best_fx = fx

        # Success counter for 1/5 rule
        successes = 0
        total_trials = 0

        call_count = 1

        while call_count < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            ys = np.empty((lam, dim))
            fys = np.empty(lam)
            for i in range(lam):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                ys[i] = y
                fys[i] = fy
                call_count += 1

            # Sort by fitness
            order = np.argsort(fys)
            fys_sorted = fys[order]
            ys_sorted = ys[order]

            # Check if any improvement
            if fys_sorted[0] < fx:
                successes += 1
            total_trials += 1

            # Update step size with 1/5 rule
            if total_trials >= 5:
                if successes / total_trials > 0.2:
                    sigma *= 1.3
                else:
                    sigma *= 0.8
                successes = 0
                total_trials = 0

            # Weighted recombination for next mean
            # Use top half
            mu = max(1, lam // 2)
            weights = np.zeros(mu)
            for i in range(mu):
                weights[i] = np.log(i + 1) - np.log(mu)
            weights /= weights.sum()

            new_x = np.zeros(dim)
            for i in range(mu):
                new_x += weights[i] * ys_sorted[i]

            # Clip to box
            new_x = np.clip(new_x, -5, 5)

            # Update best
            if fys_sorted[0] < best_fx:
                best_fx = fys_sorted[0]
                best_x = ys_sorted[0].copy()

            # Move to new mean
            x = new_x
            fx = fys_sorted[0]  # Use best of population as new reference

            # Adaptive sigma based on distance
            dist = np.linalg.norm(x - best_x)
            if dist < 1e-10:
                sigma = max(sigma * 0.5, 1e-6)

        # Final refinement: local search around best point
        if time.time() < deadline and call_count < budget:
            local_sigma = 0.1
            for _ in range(20):
                if time.time() > deadline or call_count >= budget:
                    break
                y = np.clip(best_x + local_sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                call_count += 1
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                else:
                    local_sigma *= 0.7

        return best_x, best_fx
# EVOLVE-BLOCK-END
