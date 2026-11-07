"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step-size, restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Best solution found so far
        best_x = np.zeros(dim)
        best_f = np.inf

        # Try multiple random restarts with (1+1)-ES
        # Each restart: start from a random point, run ES until budget exhausted or no improvement
        # We split the budget among restarts adaptively.

        # First, do a coarse random search to find a good starting point
        n_random = min(20, budget // 10)
        for i in range(n_random):
            if time.time() > deadline:
                break
            x = np.random.uniform(-4.5, 4.5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Now run (1+1)-ES from the best point found
        # Use a single long ES run with adaptive sigma
        x = best_x.copy()
        fx = best_f
        sigma = 1.5

        # Success counter for 1/5 rule
        successes = 0
        total_steps = 0
        max_no_improve = 50  # restart if stuck for this many steps

        no_improve_count = 0

        while total_steps < budget - n_random:
            if time.time() > deadline:
                break

            # Generate candidate
            step = sigma * np.random.standard_normal(dim)
            y = x + step
            y = np.clip(y, -5, 5)
            fy = func(y)
            total_steps += 1

            if fy < fx:
                # Accept
                x = y
                fx = fy
                successes += 1
                no_improve_count = 0

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                # Adapt sigma: increase on success
                # Use a simple rule: multiply by 1.1 on success, 0.9 on failure
                sigma *= 1.1
            else:
                no_improve_count += 1
                sigma *= 0.9

            # Clamp sigma
            sigma = max(sigma, 1e-6)
            sigma = min(sigma, 10.0)

            # If stuck for too long, try a larger jump (restart-like)
            if no_improve_count > max_no_improve:
                # Try a random perturbation
                for attempt in range(5):
                    if time.time() > deadline:
                        break
                    y = x + 2.0 * np.random.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    total_steps += 1
                    if fy < fx:
                        x = y
                        fx = fy
                        no_improve_count = 0
                        sigma = 1.0
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        break
                else:
                    # Still stuck, do a fresh random start
                    x = np.random.uniform(-4.5, 4.5, dim)
                    fx = func(x)
                    total_steps += 1
                    sigma = 2.0
                    no_improve_count = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

        # Final polish: try small steps around best_x
        if time.time() < deadline - 5:
            polish_sigma = 0.1
            for _ in range(50):
                if time.time() > deadline:
                    break
                y = best_x + polish_sigma * np.random.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    polish_sigma *= 0.8
                else:
                    polish_sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
