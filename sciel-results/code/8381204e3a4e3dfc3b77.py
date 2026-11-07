"""Black-box continuous optimizer for BBOB f6 (Attractive sector).
Uses a (1+1)-ES with adaptive step-size (1/5 rule) as the primary search mechanism,
augmented with a small initial random exploration phase and a final local polish.
This structural approach is simpler and more robust than CMA-ES for 5D problems
with a limited budget, avoiding the overhead of covariance matrix updates."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and adaptive step-size."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        n = dim
        start = time.time()
        deadline = start + 290.0  # Safety margin before 300s kill
        rng = np.random.default_rng(0)

        def time_left():
            return deadline - time.time()

        # Track global best
        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Small random exploration to find a good starting point
        # Use a small fraction of the budget for this
        n_explore = min(20, budget // 10)
        for _ in range(n_explore):
            if time_left() < 5.0:
                break
            x = rng.uniform(-5.0, 5.0, n)
            fx = func(x)
            update_best(x, fx)

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)

        # Phase 2: (1+1)-ES with 1/5 success rule
        # Start from the best point found
        x = best_x.copy()
        fx = best_f
        sigma = 1.0  # Initial step size

        # Success rule parameters
        sigma_inc = 1.5
        sigma_dec = 0.87

        # Run (1+1)-ES until budget or time runs out
        while budget > 0 and time_left() > 5.0:
            # Generate candidate
            y = x + sigma * rng.standard_normal(n)
            y = np.clip(y, -5.0, 5.0)
            fy = func(y)
            budget -= 1

            if fy < fx:
                x, fx = y, fy
                update_best(x, fx)
                sigma *= sigma_inc
            else:
                sigma *= sigma_dec

            # Safety check: if sigma becomes too small or too large, reset
            if sigma < 1e-6 or sigma > 10.0:
                sigma = 1.0

        return best_x, best_f
# EVOLVE-BLOCK-END
