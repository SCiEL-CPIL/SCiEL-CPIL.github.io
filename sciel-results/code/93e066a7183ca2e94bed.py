"""Optimizer for BBOB f23 (Katsuura) with multi-restart (1+1)-ES and time guard."""
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with time guard and adaptive sigma."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        # Multi-restart strategy: split budget among restarts
        # Use 3 restarts with decreasing budget allocations
        restarts = 3
        # Budget per restart: 60%, 25%, 15%
        budgets = [int(budget * 0.55), int(budget * 0.30), int(budget * 0.15)]
        # Ensure sum <= budget
        while sum(budgets) > budget:
            budgets[-1] -= 1
        # Ensure at least 10 per restart
        for i in range(len(budgets)):
            budgets[i] = max(10, budgets[i])

        for r in range(restarts):
            if not time_ok() or calls_made >= budget:
                break

            # Initialize: first restart at random, subsequent near best
            if r == 0:
                x = rng.uniform(-5, 5, dim)
            else:
                # Perturb best solution found so far
                if best_x is not None:
                    x = np.clip(best_x + rng.normal(0, 1.5, dim), -5, 5)
                else:
                    x = rng.uniform(-5, 5, dim)

            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            sigma = 1.5
            success_count = 0
            total_count = 0
            max_iter = min(budgets[r], budget - calls_made)

            for _ in range(max_iter):
                if not time_ok() or calls_made >= budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    success_count += 1
                    x, fx = y, fy
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    pass

                total_count += 1
                # Adapt sigma based on 1/5 rule (check every 10 steps)
                if total_count % 10 == 0:
                    rate = success_count / total_count
                    if rate > 0.2:
                        sigma *= 1.2
                    elif rate < 0.1:
                        sigma *= 0.75
                    success_count = 0
                    total_count = 0

                # Keep sigma in reasonable bounds
                sigma = np.clip(sigma, 0.001, 5.0)

            # Final sigma reset for next restart
            sigma = 1.0

        return best_x, best_f
