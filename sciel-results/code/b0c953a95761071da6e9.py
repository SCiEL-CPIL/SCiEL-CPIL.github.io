"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, deadline guard, and coordinate polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        rng = np.random.default_rng(0)

        # Initial point
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x, best_fx = x.copy(), fx
        calls = 1

        # (1+1)-ES phase
        sigma = 2.0
        no_improve = 0
        max_no_improve = 50

        while calls < self.budget:
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                no_improve = 0
                sigma *= 1.5
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
            else:
                sigma *= 0.87
                no_improve += 1

            if no_improve > max_no_improve:
                # Restart with smaller sigma and new random point
                x = rng.uniform(-5, 5, self.dim)
                fx = func(x)
                calls += 1
                sigma = 1.0
                no_improve = 0
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx

        # Coordinate-wise line search polish (Powell-like)
        if calls < self.budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_fx
            for _ in range(3):  # 3 passes
                for d in range(self.dim):
                    if time.time() > deadline or calls >= self.budget:
                        break
                    # Try small steps in each direction
                    for step in [0.5, 0.1, 0.01, 0.001]:
                        for direction in [1, -1]:
                            if calls >= self.budget or time.time() > deadline:
                                break
                            trial = x.copy()
                            trial[d] += direction * step
                            trial = np.clip(trial, -5, 5)
                            ftrial = func(trial)
                            calls += 1
                            if ftrial < fx:
                                x, fx = trial, ftrial
                                if fx < best_fx:
                                    best_x, best_fx = x.copy(), fx
                    if time.time() > deadline or calls >= self.budget:
                        break
                if time.time() > deadline or calls >= self.budget:
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
