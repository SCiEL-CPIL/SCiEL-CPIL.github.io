# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid Sobol exploration + adaptive local search for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton(self, n, dim, skip=1):
        """Generate Halton sequence points in [0,1)^dim."""
        points = np.zeros((n, dim))
        for d in range(dim):
            base = d + 2  # 2, 3, 5, 7, 11 for dim=5
            for i in range(n):
                idx = i + skip
                f = 1.0
                r = 0.0
                while idx > 0:
                    f /= base
                    r += f * (idx % base)
                    idx //= base
                points[i, d] = r
        return points

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit
        budget = self.budget
        dim = self.dim

        # Phase 1: Broad exploration with Halton sequence
        # Use first ~100 points for global coverage
        n_explore = 100
        halton_points = self._halton(n_explore, dim)
        # Map [0,1)^dim to [-5,5]^dim
        explore_points = halton_points * 10.0 - 5.0

        best_x = None
        best_f = np.inf
        calls = 0

        for i in range(n_explore):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            x = explore_points[i]
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Adaptive local search around best point
        # Use remaining budget for local refinement
        remaining = budget - calls
        if remaining <= 0 or best_x is None:
            return best_x, best_f

        # Adaptive step size local search
        sigma = 1.0
        best_x_local = best_x.copy()
        best_f_local = best_f
        rng = np.random.default_rng(42)

        # CMA-ES-like adaptation
        successes = 0
        window = 10

        for i in range(remaining):
            if time.time() > deadline:
                break
            # Generate candidate
            y = best_x_local + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1

            if fy < best_f_local:
                best_f_local = fy
                best_x_local = y.copy()
                successes += 1
                # Increase step size on success
                sigma *= 1.3
            else:
                # Decrease step size on failure
                sigma *= 0.85

            # Reset success counter periodically
            if (i + 1) % window == 0:
                if successes > window * 0.5:
                    sigma *= 1.2  # More successes, expand
                elif successes < window * 0.2:
                    sigma *= 0.7  # Few successes, contract
                successes = 0

        return best_x_local, best_f_local
# EVOLVE-BLOCK-END
