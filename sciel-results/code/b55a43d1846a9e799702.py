import numpy as np
import time


class Optimizer:
    """(1+1)-ES with restarts for BBOB-style optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # Safety margin

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls = 1

        # Restart-based (1+1)-ES
        # Each restart gets a fraction of the remaining budget
        max_restarts = 10
        restart_budget = max(20, (budget - 1) // max_restarts)

        for restart in range(max_restarts):
            if calls >= budget or time.time() >= deadline - 5:
                break

            # Initialize this restart: either continue from best or random start
            if restart == 0:
                x = best_x.copy()
                sigma = 1.5
            else:
                # Random restart with some bias toward current best
                x = best_x + rng.uniform(-3, 3, dim)
                x = np.clip(x, -5, 5)
                sigma = 2.0

            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # (1+1)-ES with 1/5 rule-like adaptation
            no_improve = 0
            for _ in range(restart_budget):
                if calls >= budget or time.time() >= deadline - 5:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1

                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.25
                    no_improve = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                    no_improve += 1

                # Early stop if stuck
                if no_improve > 15:
                    break

            # If we didn't improve much, restart
            if fx >= best_f and no_improve > 10:
                continue

        return best_x, best_f
