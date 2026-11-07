"""Multi-start adaptive (1+1)-ES with deadline guard for BBOB Weierstrass (f16)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start adaptive (1+1)-ES for the BBOB Weierstrass (f16) task."""

    def __init__(self, budget, dim):
        self.budget, self.dim = int(budget), int(dim)

    def __call__(self, func):
        dim = self.dim
        rng = np.random.default_rng(42)
        deadline = time.time() + 285.0  # safety margin below the 330s kill

        # ---- Phase 1: cheap multi-start to find a good basin early ----
        # Use a small number of diverse seeds to get early progress
        n_starts = 6
        seeds = [np.zeros(dim)]
        for _ in range(n_starts - 1):
            seeds.append(rng.uniform(-4.0, 4.0, dim))

        best_x = None
        best_f = np.inf
        for s in seeds:
            if time.time() > deadline:
                break
            x_try = np.clip(s, -5.0, 5.0)
            fx = func(x_try)
            if fx < best_f:
                best_f = fx
                best_x = x_try.copy()

        if best_x is None:
            x0 = np.zeros(dim)
            best_f = func(x0)
            best_x = x0.copy()

        # ---- Phase 2: adaptive (1+1)-ES from the best start ----
        x = best_x.copy()
        fx = best_f
        sigma = 2.0  # start large to explore, then adapt down
        min_sigma, max_sigma = 1e-7, 8.0
        target = 1.0 / 5.0
        c_sigma = 0.35  # faster adaptation

        n_calls = len(seeds)
        remaining = self.budget - n_calls
        for i in range(remaining):
            if time.time() > deadline:
                break
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            fy = func(y)
            if fy < fx:
                x, fx = y, fy
                sigma *= np.exp(c_sigma * (1.0 - target))
            else:
                sigma *= np.exp(c_sigma * (0.0 - target))
            sigma = min(max(sigma, min_sigma), max_sigma)

        return x, fx
# EVOLVE-BLOCK-END
