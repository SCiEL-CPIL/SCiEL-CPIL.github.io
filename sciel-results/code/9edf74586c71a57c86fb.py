"""Rewritten optimizer for BBOB-style f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with time guard and adaptive step size."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit, 10s safety margin

        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        sigma = 1.0
        n_success = 0
        n_total = 0
        success_window = 20

        # Phase 1: Exploration with (1+1)-ES
        for i in range(self.budget - 1):
            if time.time() > deadline:
                break

            # Adaptive sigma based on success rate
            if n_total >= success_window:
                success_rate = n_success / n_total
                if success_rate > 0.5:
                    sigma *= 1.2
                elif success_rate < 0.3:
                    sigma *= 0.8
                n_success = 0
                n_total = 0

            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
                sigma *= 1.3
                n_success += 1
            else:
                sigma *= 0.9
            n_total += 1

        # Phase 2: Fine-tuning with smaller sigma
        remaining = self.budget - calls
        if remaining > 0 and time.time() < deadline:
            sigma = max(0.01, sigma * 0.1)
            for i in range(remaining):
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_fx:
                        best_x, best_fx = x.copy(), fx
                    sigma *= 1.1
                else:
                    sigma *= 0.95

        return best_x, best_fx
# EVOLVE-BLOCK-END
