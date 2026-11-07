"""Optimized program for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(mu+lambda)-ES with adaptive step size for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 290s safety margin before 330s kill

        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        # Population size
        lam = 12  # offspring per generation
        mu = max(1, lam // 2)  # number of selected parents

        # Initialize
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        sigma = 1.5
        sigma_min = 1e-6
        sigma_max = 10.0

        # Track for stagnation detection
        no_improve_count = 0
        restart_threshold = 60  # restart if no improvement for this many evals

        evals_used = 1
        gen = 0

        while evals_used < budget:
            # Check time
            if time.time() > deadline - 5:
                break

            # Generate offspring
            offspring = []
            for i in range(lam):
                if time.time() > deadline - 5:
                    break
                child = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                child_fx = func(child)
                offspring.append((child_fx, child))
                evals_used += 1
                if evals_used >= budget:
                    break

            if not offspring:
                break

            # Sort offspring by fitness
            offspring.sort(key=lambda t: t[0])

            # Update best
            for f_val, x_val in offspring:
                if f_val < best_fx:
                    best_fx = f_val
                    best_x = x_val.copy()

            # Select top mu
            if len(offspring) >= mu:
                selected = [x_val for f_val, x_val in offspring[:mu]]
                # Weighted recombination (equal weights for simplicity)
                x_new = np.mean(selected, axis=0)

                # Check if we improved
                f_new = func(x_new)
                evals_used += 1
                if evals_used >= budget:
                    break

                if f_new < fx:
                    # Success: increase sigma
                    sigma = min(sigma * 1.3, sigma_max)
                    no_improve_count = 0
                else:
                    # Failure: decrease sigma
                    sigma = max(sigma * 0.8, sigma_min)
                    no_improve_count += 1

                # Move to new center
                x = x_new
                fx = f_new

            # Stagnation restart
            if no_improve_count > restart_threshold:
                # Try a random restart near current best
                x = np.clip(best_x + sigma * 0.5 * rng.standard_normal(dim), -5, 5)
                fx = func(x)
                evals_used += 1
                no_improve_count = 0
                sigma = max(sigma * 2.0, 0.5)
                if evals_used >= budget:
                    break

            gen += 1

        return best_x, best_fx
# EVOLVE-BLOCK-END
