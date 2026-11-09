"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-scale coordinate descent with adaptive steps for Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        # Phase 1: Multi-scale coordinate descent
        # Büche-Rastrigin has a specific structure: f(x) = 10*dim + sum(f_buch(x_i) - 10)
        # where f_buch(x) = (x-2)^2 * (sin((x-2)^2) + 1) + 10
        # The global minimum is at x_i = 2 for all i (before shift/rotation).
        # After shift and rotation, the minimum is somewhere in [-4, 4]^5.
        # 
        # Strategy: Use coordinate descent with multiple step sizes, starting large
        # and shrinking. This exploits the separable-like structure even under rotation
        # by systematically scanning each dimension.

        x = np.zeros(self.dim)
        fx = func(x)
        calls = 1

        # Multi-scale steps: geometric sequence from large to small
        base_steps = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]

        for step in base_steps:
            if calls >= self.budget:
                break
            if time.time() > deadline:
                break
            
            improved = True
            while improved and calls < self.budget and time.time() < deadline:
                improved = False
                for d in range(self.dim):
                    if calls >= self.budget or time.time() > deadline:
                        break
                    # Try +step and -step in dimension d
                    for direction in [1, -1]:
                        if calls >= self.budget or time.time() > deadline:
                            break
                        y = x.copy()
                        y[d] = np.clip(y[d] + direction * step, -5, 5)
                        fy = func(y)
                        calls += 1
                        if fy < fx:
                            x = y
                            fx = fy
                            improved = True

        # Phase 2: Fine-tuning with small random perturbations
        if calls < self.budget and time.time() < deadline:
            rng = np.random.default_rng(42)
            sigma = 0.01
            for _ in range(self.budget - calls):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x = y
                    fx = fy
                    sigma = min(sigma * 1.2, 0.1)
                else:
                    sigma *= 0.95

        return x, fx
# EVOLVE-BLOCK-END
