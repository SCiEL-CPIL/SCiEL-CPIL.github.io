"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, stall detection, and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # 30s safety margin before 300s limit
        rng = np.random.default_rng(0)

        # Phase 1: Random restarts to find a good starting point
        best_x = None
        best_fx = np.inf
        for i in range(10):
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            if time.time() > deadline:
                break

        x = best_x.copy()
        fx = best_fx
        sigma = 1.0
        stall_count = 0
        max_stall = 30

        for _ in range(self.budget - 10):
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3
                stall_count = 0
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                sigma *= 0.9
                stall_count += 1

            # Stall detection: restart from a perturbation of best point
            if stall_count >= max_stall:
                stall_count = 0
                sigma = max(sigma * 0.5, 0.1)
                # Try a few local perturbations around best
                for _ in range(5):
                    y = np.clip(best_x + sigma * rng.standard_normal(self.dim), -5, 5)
                    fy = func(y)
                    if fy < fx:
                        x, fx = y, fy
                        break
                    if fy < best_fx:
                        best_fx = fy
                        best_x = y.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
