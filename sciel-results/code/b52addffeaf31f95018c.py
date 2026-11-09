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

        def elapsed_ok():
            return (time.time() - start_time) < deadline

        # Phase 1: Multi-start exploration
        # Evaluate several random points and keep the best
        n_starts = 10
        best_x = None
        best_fx = np.inf
        calls_used = 0

        for _ in range(n_starts):
            if calls_used >= budget - 1 or not elapsed_ok():
                break
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            calls_used += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if best_x is None:
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            calls_used += 1
            best_x = x.copy()
            best_fx = fx

        # Phase 2: (1+1)-ES exploitation from the best start
        x = best_x.copy()
        fx = best_fx
        sigma = 1.0

        # Adaptive sigma with 1/5 rule-like behavior
        # Track success rate over a window
        window = 20
        successes = 0
        total_steps = 0

        while calls_used < budget - 1 and elapsed_ok():
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            calls_used += 1

            if fy < fx:
                x, fx = y, fy
                successes += 1
                sigma *= 1.05
            else:
                sigma *= 0.95

            total_steps += 1
            if total_steps >= window:
                rate = successes / total_steps
                if rate > 0.2:
                    sigma *= 1.1
                elif rate < 0.1:
                    sigma *= 0.9
                successes = 0
                total_steps = 0

            # Keep sigma in reasonable bounds
            sigma = max(sigma, 1e-6)
            sigma = min(sigma, 10.0)

        # Phase 3: Final polish with very small steps
        if calls_used < budget - 1 and elapsed_ok():
            sigma = 0.01
            polish_steps = min(50, budget - 1 - calls_used)
            for _ in range(polish_steps):
                if not elapsed_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy

        return x, fx
# EVOLVE-BLOCK-END
