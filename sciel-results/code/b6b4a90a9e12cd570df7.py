"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, restarts, and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        rng = np.random.default_rng(0)
        best_x = None
        best_fx = np.inf
        calls = 0

        # Phase 1: CMA-like restarts with (1+1)-ES
        # Use multiple restarts to escape local minima of Weierstrass
        max_restarts = 8
        calls_per_restart = max(1, (self.budget - 1) // max_restarts)
        
        for restart in range(max_restarts):
            if calls >= self.budget - 1:
                break
            if time.time() > deadline:
                break

            # Initialize with diversity
            if restart == 0:
                x = rng.uniform(-5, 5, self.dim)
            else:
                # Perturb around best found so far
                if best_x is not None:
                    x = np.clip(best_x + rng.normal(0, 1.5, self.dim), -5, 5)
                else:
                    x = rng.uniform(-5, 5, self.dim)

            fx = func(x)
            calls += 1

            if fx < best_fx:
                best_x, best_fx = x.copy(), fx

            sigma = 1.0
            no_improve = 0

            for _ in range(calls_per_restart):
                if calls >= self.budget - 1:
                    break
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    no_improve = 0
                    if fx < best_fx:
                        best_x, best_fx = x.copy(), fx
                else:
                    sigma *= 0.85
                    no_improve += 1

                # Early restart if stuck
                if no_improve > 30:
                    break

        # Phase 2: Fine-tuning around best found
        if best_x is not None and calls < self.budget:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.1
            remaining = self.budget - calls
            for _ in range(remaining):
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    if fx < best_fx:
                        best_x, best_fx = x.copy(), fx
                else:
                    sigma *= 0.9

        return best_x, best_fx
# EVOLVE-BLOCK-END
