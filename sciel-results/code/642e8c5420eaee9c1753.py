"""Optimizer for BBOB f14 (Different powers) with hidden shift/rotation/offset."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """(1+1)-ES with 1/5 success rule and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin

        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        # --- Phase 1: Initialization ---
        # Use a mix of random points and the origin to ensure good coverage
        n_init = min(30, budget // 5)
        best_x = None
        best_f = np.inf

        # Evaluate origin first (often a good starting point if shift is small)
        x0 = np.zeros(dim)
        f0 = float(func(x0))
        best_x = x0.copy()
        best_f = f0
        calls_used = 1

        # Evaluate random points
        init_points = rng.uniform(-5, 5, size=(n_init, dim))
        for i in range(n_init):
            if time.time() > deadline or calls_used >= budget:
                break
            x = init_points[i]
            fx = float(func(x))
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # --- Phase 2: (1+1)-ES ---
        x = best_x.copy()
        fx = best_f
        
        # Initialize sigma based on problem scale or previous spread
        # A moderate starting sigma is usually safe
        sigma = 1.0
        
        # 1/5 rule parameters
        window_size = 20
        successes = []

        for gen in range(budget - calls_used):
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break

            # Generate offspring
            step = sigma * rng.standard_normal(dim)
            y = np.clip(x + step, -5, 5)
            fy = float(func(y))
            calls_used += 1

            # Update success history
            success = fy < fx
            successes.append(success)
            if len(successes) > window_size:
                successes.pop(0)

            if success:
                x = y
                fx = fy

            # Adapt sigma
            if len(successes) >= 5:
                rate = np.mean(successes)
                if rate > 0.2:
                    sigma *= 1.1
                elif rate < 0.2:
                    sigma *= 0.9
                sigma = np.clip(sigma, 0.01, 10.0)

        return x, fx
# EVOLVE-BLOCK-END
