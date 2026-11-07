"""Refined (1+1)-ES with improved sigma adaptation and no NM polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start adaptive (1+1)-ES for the BBOB Weierstrass (f16) task."""

    def __init__(self, budget, dim):
        self.budget, self.dim = int(budget), int(dim)

    def __call__(self, func):
        dim = self.dim
        rng = np.random.default_rng(12345)
        start_time = time.time()
        deadline = start_time + 290.0  # tighter safety margin
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            if call_count >= self.budget:
                return np.inf
            call_count += 1
            return func(np.clip(x, -5.0, 5.0))

        # ---- Phase 1: multi-start with structured + random seeds ----
        # Include center, axis-aligned points, and randoms for diverse basins
        seeds = [np.zeros(dim)]
        # Add points along each axis at moderate distance
        for i in range(dim):
            p = np.zeros(dim)
            p[i] = 2.0
            seeds.append(p)
            p2 = np.zeros(dim)
            p2[i] = -2.0
            seeds.append(p2)
        # Random starts
        for _ in range(6):
            seeds.append(rng.uniform(-4.5, 4.5, dim))

        # De-duplicate
        seen = []
        uniq = []
        for s in seeds:
            key = tuple(np.round(s, 6))
            if key not in seen:
                seen.append(key)
                uniq.append(s)

        best_x = None
        best_f = np.inf
        for s in uniq:
            if time.time() > deadline and best_x is not None:
                break
            fx = safe_func(s)
            if fx < best_f:
                best_f = fx
                best_x = s.copy()

        if best_x is None:
            return np.zeros(dim), np.inf

        # ---- Phase 2: adaptive (1+1)-ES with improved sigma control ----
        x = best_x.copy()
        fx = best_f
        sigma = 0.5  # start smaller for oscillatory function
        min_sigma, max_sigma = 1e-7, 5.0
        c_sigma = 0.3  # faster adaptation
        target = 1.0 / (dim + 1)  # theoretical optimal success rate

        # Track recent successes for smoother adaptation
        successes = []
        window = 10

        remaining = self.budget - call_count
        for step in range(remaining):
            if time.time() > deadline:
                break
            # Occasionally do a larger jump to escape local minima
            if step % 50 == 0 and sigma < 0.1:
                sigma = 0.3

            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            fy = safe_func(y)
            improved = fy < fx
            if improved:
                x, fx = y, fy

            # Update success history
            successes.append(1.0 if improved else 0.0)
            if len(successes) > window:
                successes.pop(0)

            # Adapt sigma based on recent success rate
            recent_rate = np.mean(successes)
            sigma *= np.exp(c_sigma * (recent_rate - target))
            sigma = min(max(sigma, min_sigma), max_sigma)

        return x, fx
# EVOLVE-BLOCK-END
