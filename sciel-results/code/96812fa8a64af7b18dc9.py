"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start with (1+1)-ES refinement for Rastrigin-rotated f15."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 270.0  # 30s safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls = 0

        def safe_func(x):
            nonlocal calls
            calls += 1
            return func(x)

        def time_ok():
            return time.time() < deadline

        # Phase 1: Multi-start with quick coordinate descent
        # 12 diverse starts, each gets ~15 calls for coordinate search
        n_starts = 12
        starts = [np.zeros(dim)]
        for i in range(n_starts - 1):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        for x0 in starts:
            if calls >= budget or not time_ok():
                break

            x = np.clip(x0.copy(), -5, 5)
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Quick coordinate-wise descent: 2 passes, dim*2 calls max
            step = np.full(dim, 1.0)
            for _pass in range(2):
                if calls >= budget or not time_ok():
                    break
                for d in range(dim):
                    if calls >= budget or not time_ok():
                        break
                    # Try +
                    xt = x.copy()
                    xt[d] = np.clip(xt[d] + step[d], -5, 5)
                    if not np.allclose(xt, x):
                        ft = safe_func(xt)
                        if ft < fx:
                            x, fx = xt, ft
                            step[d] *= 1.5
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            continue
                    # Try -
                    xt = x.copy()
                    xt[d] = np.clip(xt[d] - step[d], -5, 5)
                    if not np.allclose(xt, x):
                        ft = safe_func(xt)
                        if ft < fx:
                            x, fx = xt, ft
                            step[d] *= 1.5
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                step *= 0.5

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)

        # Phase 2: (1+1)-ES with random directions on best solution
        # Use remaining budget for ES refinement
        x = best_x.copy()
        fx = best_f
        sigma = 1.5
        remaining = budget - calls

        for _ in range(remaining):
            if calls >= budget or not time_ok():
                break

            # Random direction in dim-space (handles rotation)
            direction = rng.standard_normal(dim)
            direction /= (np.linalg.norm(direction) + 1e-12)

            y = np.clip(x + sigma * direction, -5, 5)
            fy = safe_func(y)

            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * 1.3, 4.0)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma = max(sigma * 0.85, 0.01)

        # Phase 3: Fine polish with smaller sigma if budget remains
        remaining2 = budget - calls
        if remaining2 > 5 and time_ok():
            sigma = 0.1
            for _ in range(min(remaining2, 20)):
                if calls >= budget or not time_ok():
                    break
                direction = rng.standard_normal(dim)
                direction /= (np.linalg.norm(direction) + 1e-12)
                y = np.clip(x + sigma * direction, -5, 5)
                fy = safe_func(y)
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
