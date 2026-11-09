"""Optimizer for BBOB f14 (Different powers) with hidden shift/rotation/offset."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with robust initialization, 1/5-rule adaptation, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # --- Phase 1: Robust Initialization ---
        # Evaluate origin and a few random points to get a good starting point
        candidates = [np.zeros(dim)]
        for _ in range(4):
            candidates.append(rng.uniform(-5, 5, dim))

        best_x = None
        best_f = np.inf
        for c in candidates:
            fc = float(func(c))
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        calls_used = len(candidates)
        remaining = budget - calls_used

        if remaining <= 0:
            return best_x, best_f

        # --- Phase 2: (1+1)-ES with 1/5 success rule ---
        x = best_x.copy()
        fx = best_f
        sigma = 0.5  # moderate starting step size

        # 1/5 rule parameters
        window_size = 20
        successes = []

        for i in range(remaining):
            # Time guard
            if time.time() > deadline:
                break

            # Generate mutant
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
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

            # Adapt sigma using 1/5 success rule
            if len(successes) >= 5:
                rate = np.mean(successes)
                if rate > 0.2:
                    sigma *= 1.1
                elif rate < 0.2:
                    sigma *= 0.9
                sigma = np.clip(sigma, 1e-6, 10.0)

        return x, fx
# EVOLVE-BLOCK-END
