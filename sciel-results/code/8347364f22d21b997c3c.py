"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-like optimizer with adaptive step size for BBOB f20 (Schwefel)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # Initial point: center of the box is a good starting point for Schwefel
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Initial step size
        sigma = 1.5

        # Population size for CMA-ES-like sampling
        lam = 4 + int(3 * np.log(dim))  # ~ 4 + 3*ln(5) ≈ 4 + 4.83 = 9
        lam = max(4, min(20, lam))

        # Track success rate for 1/5 rule
        success_count = 0
        last_sigma = sigma

        # We'll use a mix: each iteration we generate lam candidates and pick the best
        # This gives more exploration than (1+1)-ES

        it = 0
        total_calls = 1  # already made one call

        while total_calls < budget:
            if time.time() >= deadline:
                break

            # Generate lam candidates
            candidates = []
            for _ in range(lam):
                if total_calls >= budget:
                    break
                if time.time() >= deadline:
                    break
                y = np.clip(x + sigma * np.random.randn(dim), -5, 5)
                fy = func(y)
                total_calls += 1
                candidates.append((fy, y))

            if not candidates:
                break

            # Sort by fitness
            candidates.sort(key=lambda t: t[0])
            best_fy, best_y = candidates[0]

            if best_fy < fx:
                # Accept the best candidate
                x = best_y
                fx = best_fy
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                success_count += 1
            else:
                success_count += 0

            # Update sigma based on success rate
            # Simple adaptation: if we found an improvement, increase sigma; else decrease
            # Use a softer rule
            if best_fy < fx + 1e-10:  # was an improvement
                sigma *= 1.2
            else:
                sigma *= 0.9

            # Clamp sigma
            sigma = max(1e-6, min(10.0, sigma))

            # Occasionally do a larger exploration step (restart-like)
            if it % 50 == 0 and it > 0:
                # Try a few random points to escape local minima
                for _ in range(3):
                    if total_calls >= budget:
                        break
                    if time.time() >= deadline:
                        break
                    ry = np.random.uniform(-5, 5, dim)
                    rfy = func(ry)
                    total_calls += 1
                    if rfy < fx:
                        x = ry
                        fx = rfy
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                    elif rfy < best_fx:
                        # Even if not better than current x, track global best
                        if rfy < best_fx:
                            best_fx = rfy
                            best_x = ry.copy()

            it += 1

        return best_x, best_fx
# EVOLVE-BLOCK-END
