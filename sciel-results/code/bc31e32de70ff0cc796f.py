"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with origin init, multiplicative sigma adaptation, and multiple restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 315.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Best solution found overall
        best_x = None
        best_f = np.inf
        eval_count = 0

        def try_update(x, fx):
            nonlocal best_x, best_f, eval_count
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 1: Main (1+1)-ES from origin
        x = np.zeros(n)
        fx = func(x)
        try_update(x, fx)

        sigma = 2.0
        sigma_min = 1e-8
        sigma_max = 50.0

        # Multiple restarts with decreasing no_improve threshold
        max_restarts = 3
        for restart in range(max_restarts):
            # Run (1+1)-ES
            sigma = 2.0 if restart == 0 else 1.0
            no_improve = 0
            max_no_improve = 80 + restart * 30

            for i in range(budget - eval_count):
                if eval_count >= budget:
                    break
                if time.time() > deadline - 3.0:
                    break

                z = rng.standard_normal(n)
                y = np.clip(x + sigma * z, -5, 5)
                fy = func(y)
                eval_count += 1

                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = min(sigma * 1.3, sigma_max)
                    no_improve = 0
                else:
                    sigma *= 0.85
                    no_improve += 1

                if sigma < sigma_min or no_improve >= max_no_improve:
                    break

            # If we used all budget or time is up, stop
            if eval_count >= budget or time.time() > deadline - 3.0:
                break

            # Restart: try from a perturbation of best or a random point
            remaining = budget - eval_count
            if remaining < 5:
                break

            if best_x is not None and rng.random() < 0.6:
                # Perturb best solution
                x = np.clip(best_x + rng.standard_normal(n) * (0.5 + 0.5 * rng.random()), -5, 5)
            else:
                # Try origin or a random point
                if rng.random() < 0.5:
                    x = np.zeros(n)
                else:
                    x = rng.uniform(-3, 3, n)

            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
