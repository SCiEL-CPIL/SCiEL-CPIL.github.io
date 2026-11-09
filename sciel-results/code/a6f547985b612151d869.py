"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES core with initial exploration and final coordinate polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        calls = [0]

        def safe_func(x):
            if calls[0] >= budget:
                raise StopIteration
            calls[0] += 1
            return float(func(np.asarray(x, dtype=np.float64)))

        rng = np.random.default_rng(0)

        # ---- Phase 1: Brief multi-start exploration (use ~10% of budget) ----
        explore_budget = max(10, budget // 10)
        best_x = None
        best_f = np.inf

        starts = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
        ]

        for s in starts:
            if calls[0] >= explore_budget:
                break
            x = np.clip(s, -5, 5)
            try:
                fx = safe_func(x)
            except StopIteration:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # ---- Phase 2: (1+1)-ES main search (dominant phase) ----
        x = best_x.copy()
        fx = best_f
        sigma = 1.5

        # Use remaining budget for (1+1)-ES
        remaining = budget - calls[0]
        for _ in range(remaining):
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            try:
                fy = safe_func(y)
            except StopIteration:
                break

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.4
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma *= 0.88

            if sigma < 1e-8:
                break

        # ---- Phase 3: Final coordinate descent polish (use remaining budget) ----
        if calls[0] < budget and time.time() < deadline:
            center = best_x.copy()
            cf = best_f
            step = 0.1

            for _ in range(15):
                if calls[0] >= budget or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    if calls[0] >= budget or time.time() > deadline:
                        break
                    for sign in (1, -1):
                        if calls[0] >= budget:
                            break
                        x2 = center.copy()
                        x2[d] = np.clip(center[d] + sign * step, -5, 5)
                        try:
                            f2 = safe_func(x2)
                        except StopIteration:
                            break
                        if f2 < cf:
                            cf = f2
                            center = x2
                            improved = True
                    if time.time() > deadline:
                        break
                if improved:
                    step *= 0.7
                else:
                    step *= 0.5
                if step < 1e-7:
                    break

            if cf < best_f:
                best_f = cf
                best_x = center.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
