"""Hybrid global-local optimizer for BBOB f7 (step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid: coarse grid + random restarts + adaptive (1+1)-ES refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 290.0  # 40s safety margin before 330s kill
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(12345)

        # Phase 1: coarse grid search to find a good basin (step ellipsoidal
        # has a large flat region; locating the center is the key move).
        # 2^5 = 32 points, cheap and gives a solid starting point.
        grid_vals = np.linspace(-4.5, 4.5, 2)
        best_x = None
        best_f = np.inf
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        for m in range(2):
                            if best_x is not None:
                                if time.time() > deadline:
                                    return best_x, best_f
                            pt = np.array([grid_vals[i], grid_vals[j], grid_vals[k],
                                          grid_vals[l], grid_vals[m]], dtype=float)
                            f = func(pt)
                            if f < best_f:
                                best_f = f
                                best_x = pt.copy()

        # Phase 2: random restarts to explore other basins briefly.
        for _ in range(12):
            if time.time() > deadline:
                break
            pt = rng.uniform(-4.5, 4.5, dim)
            f = func(pt)
            if f < best_f:
                best_f = f
                best_x = pt.copy()

        # Phase 3: adaptive (1+1)-ES with 1/5 success rule, but with
        # time-aware early exit. Use a generous initial sigma.
        x = best_x.copy()
        fx = best_f
        sigma = 1.5
        # Track success rate over a window to adapt sigma
        successes = 0
        window = 20
        trials_in_window = 0

        remaining = budget - (32 + 12)
        for _ in range(remaining):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            trials_in_window += 1
            if fy < fx:
                x = y
                fx = fy
                successes += 1
            if trials_in_window >= window:
                rate = successes / trials_in_window
                if rate > 0.3:
                    sigma *= 1.2
                elif rate < 0.15:
                    sigma *= 0.7
                else:
                    sigma *= 0.95
                successes = 0
                trials_in_window = 0
            sigma = max(sigma, 1e-6)

        return x, fx
# EVOLVE-BLOCK-END
