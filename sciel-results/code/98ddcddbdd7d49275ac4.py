"""Optimizer for BBOB f14 (Different powers) with hidden shift/rotation/offset."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """(1+1)-ES with 1/5 success rule, robust initialization, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill

        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        # --- Phase 1: Robust Initialization ---
        # Evaluate origin and a few random points to get a good starting point
        n_init = min(20, budget // 5)
        candidates = [np.zeros(dim)]
        for _ in range(n_init - 1):
            candidates.append(rng.uniform(-5, 5, dim))

        best_x = None
        best_f = np.inf
        for c in candidates:
            if time.time() > deadline:
                break
            fc = float(func(c))
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        calls_used = sum(1 for _ in candidates)  # approximate, but we track below
        # Actually, let's track calls properly
        calls_used = 0
        best_x = None
        best_f = np.inf
        for c in candidates:
            if time.time() > deadline or calls_used >= budget:
                break
            fc = float(func(c))
            calls_used += 1
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        remaining = budget - calls_used
        if remaining <= 0 or best_x is None:
            return best_x, best_f

        # --- Phase 2: (1+1)-ES with 1/5 success rule ---
        x = best_x.copy()
        fx = best_f
        sigma = 0.5  # moderate starting step size

        # 1/5 rule parameters
        window_size = 20
        successes = []

        for gen in range(remaining):
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
