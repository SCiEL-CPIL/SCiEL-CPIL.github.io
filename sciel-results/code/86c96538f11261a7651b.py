"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive step size for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin

        # Multi-start: evaluate several random points to find a good basin
        # Use first ~20% of budget for exploration, rest for exploitation
        explore_budget = max(10, budget // 5)
        best_x = None
        best_f = float('inf')

        # Phase 1: Multi-start random sampling
        for i in range(explore_budget):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = rng.uniform(-5, 5, n)
            best_f = func(best_x)

        # Phase 2: Local optimization from best starting point
        x = best_x.copy()
        fx = best_f
        sigma = 1.5

        # Adaptive (1+1)-ES from the best basin
        remaining_budget = budget - explore_budget
        for i in range(remaining_budget):
            if time.time() > deadline:
                break

            # Generate trial point
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3  # Expand on success
            else:
                sigma *= 0.85  # Contract on failure

            # Occasionally try a larger jump to escape local minima
            if i % 50 == 49 and sigma < 0.1:
                sigma = 1.0

        return x, fx
# EVOLVE-BLOCK-END
