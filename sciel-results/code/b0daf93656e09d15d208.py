import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and restart mechanism for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_fx = float('inf')
        total_evals = 0

        def evaluate(x):
            nonlocal total_evals, best_x, best_fx
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            total_evals += 1
            if f < best_fx:
                best_fx = f
                best_x = x.copy()
            return f

        # Phase 1: Multiple short (1+1)-ES runs with different starting points
        # Use ~80% of budget for exploration with restarts
        phase1_budget = int(budget * 0.8)
        num_restarts = 3
        per_restart_budget = phase1_budget // num_restarts

        for r in range(num_restarts):
            if total_evals >= phase1_budget:
                break
            if time.time() > deadline:
                break

            # Different starting points for each restart
            if r == 0:
                x = rng.uniform(-5, 5, dim)
            elif r == 1:
                x = np.zeros(dim) + rng.normal(0, 0.5, dim)
            else:
                x = rng.uniform(-2, 2, dim)

            fx = evaluate(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            sigma = 2.0
            max_sigma = 5.0
            min_sigma = 1e-6
            stagnant = 0

            for step in range(per_restart_budget):
                if total_evals >= phase1_budget:
                    break
                if time.time() > deadline:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    # Increase sigma on success (1/5 rule: multiply by ~1.348 for 20% success rate in high dim)
                    sigma *= 1.348
                    if sigma > max_sigma:
                        sigma = max_sigma
                    stagnant = 0
                else:
                    # Decrease sigma on failure
                    sigma *= 0.8
                    if sigma < min_sigma:
                        sigma = min_sigma
                    stagnant += 1

                # If very stagnant, break early to save budget for next restart
                if stagnant > 50 and sigma < 0.01:
                    break

        # Phase 2: Fine-tuning from best found
        remaining_budget = budget - total_evals
        if remaining_budget > 0 and best_x is not None:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.1
            max_sigma = 1.0
            min_sigma = 1e-8

            for _ in range(remaining_budget):
                if time.time() > deadline:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    sigma *= 1.2
                    if sigma > max_sigma:
                        sigma = max_sigma
                else:
                    sigma *= 0.85
                    if sigma < min_sigma:
                        sigma = min_sigma

        return best_x, best_fx
