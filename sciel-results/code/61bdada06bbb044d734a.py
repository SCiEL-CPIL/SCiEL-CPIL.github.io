"""Rewritten optimizer for BBOB-style Rastrigin f3."""
import numpy as np
import time


class Optimizer:
    """(1+lambda)-ES with adaptive step size and restarts for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        best_x = None
        best_f = np.inf

        # We'll use a (1+lambda) strategy with restarts
        # For Rastrigin in 5D, a population-based approach helps escape local minima

        lambda_ = 10  # offspring count
        budget_remaining = self.budget
        rng = np.random.default_rng(42)

        # Initial point
        x = rng.uniform(-4, 4, self.dim)
        fx = func(x)
        budget_remaining -= 1
        best_x = x.copy()
        best_f = fx

        sigma = 1.0
        sigma_min = 1e-4
        sigma_max = 5.0

        # Track for adaptation
        success = 0
        total = 0
        target_success = 0.2  # 1/5 rule generalized

        while budget_remaining > 0:
            if time.time() > deadline:
                break

            # Generate offspring
            offspring = []
            for _ in range(min(lambda_, budget_remaining)):
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                budget_remaining -= 1
                offspring.append((y, fy))

            if not offspring:
                break

            # Select best offspring
            offspring.sort(key=lambda t: t[1])
            best_y, best_fy = offspring[0]

            # Update sigma based on success
            if best_fy < fx:
                x = best_y
                fx = best_fy
                success += 1
                if fx < best_f:
                    best_x = x.copy()
                    best_f = fx
            else:
                # Keep current x but update sigma
                pass

            total += 1
            # Adapt sigma
            if total >= 5:
                rate = success / total
                if rate > target_success:
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                sigma = np.clip(sigma, sigma_min, sigma_max)
                success = 0
                total = 0

            # Check for restart: if no improvement in last 50 evaluations
            if budget_remaining > 0:
                # Simple restart heuristic: if sigma is very small or no progress
                if sigma < sigma_min * 2 and budget_remaining > 20:
                    # Restart from a new random point
                    x = rng.uniform(-4, 4, self.dim)
                    fx = func(x)
                    budget_remaining -= 1
                    sigma = 1.0
                    if fx < best_f:
                        best_x = x.copy()
                        best_f = fx

        # Final polish: coordinate descent / local refinement
        if budget_remaining > 0 and time.time() < deadline:
            # Try a few local refinements around best_x
            for _ in range(min(10, budget_remaining)):
                if time.time() > deadline:
                    break
                # Coordinate descent
                improved = False
                for d in range(self.dim):
                    for step in [0.1, 0.01, 0.001]:
                        for sign in [1, -1]:
                            y = best_x.copy()
                            y[d] = np.clip(y[d] + sign * step, -5, 5)
                            fy = func(y)
                            budget_remaining -= 1
                            if fy < best_f:
                                best_x = y
                                best_f = fy
                                improved = True
                            if budget_remaining <= 0:
                                break
                        if budget_remaining <= 0:
                            break
                    if budget_remaining <= 0:
                        break
                if not improved and budget_remaining <= 0:
                    break

        return best_x, best_f
