"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer for BBOB-style f17 (Schaffers F7) in 5D.
    Strategy:
      1. Coarse random search to find a good initial point.
      2. CMA-ES style adaptive step-size evolution strategy for exploitation.
      3. Time-aware loop to stay within budget.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Coarse random search (use ~20% of budget)
        n_random = max(10, budget // 5)
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        for i in range(n_random - 1):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Phase 2: (1+1)-ES with adaptive sigma
        x = best_x.copy()
        fx = best_fx
        sigma = 1.0
        sigma_min = 1e-6
        sigma_max = 10.0
        success_count = 0
        total_trials = 0
        target_success_rate = 1/5

        remaining_budget = budget - n_random
        for i in range(remaining_budget):
            if time.time() > deadline:
                break

            # Generate trial point
            direction = rng.standard_normal(dim)
            y = np.clip(x + sigma * direction, -5, 5)
            fy = func(y)
            total_trials += 1

            if fy < fx:
                x, fx = y, fy
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                success_count += 1
                sigma = min(sigma * 1.2, sigma_max)
            else:
                sigma = max(sigma * 0.8, sigma_min)

            # Adapt sigma based on success rate every 10 steps
            if total_trials % 10 == 0 and total_trials >= 10:
                recent_success_rate = success_count / total_trials
                if recent_success_rate > target_success_rate:
                    sigma = min(sigma * 1.1, sigma_max)
                elif recent_success_rate < target_success_rate * 0.5:
                    sigma = max(sigma * 0.9, sigma_min)
                success_count = 0
                total_trials = 0

        return best_x, best_fx
# EVOLVE-BLOCK-END
