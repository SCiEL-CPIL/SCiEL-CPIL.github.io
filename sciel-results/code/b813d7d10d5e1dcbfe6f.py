"""Rewritten optimizer: multi-start CMA-ES with L-BFGS-B polish, deadline-guarded."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start CMA-ES with L-BFGS-B polishing for Lunacek bi-Rastrigin f24."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._calls = 0
        self._deadline = None

    def __call__(self, func):
        self._calls = 0
        self._deadline = time.time() + 290.0  # safety margin under 300s

        dim = self.dim
        budget = self.budget
        calls = [0]

        def f(x):
            if calls[0] >= budget:
                return 1e12
            calls[0] += 1
            val = func(np.asarray(x, dtype=float))
            return float(val)

        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf

        # Phase 1: Multi-start CMA-ES (lightweight, no scipy overhead per step)
        # We allocate most budget to exploration, then polish.
        n_starts = 8
        per_start_budget = max(20, (budget - 100) // n_starts)

        for s in range(n_starts):
            if calls[0] >= budget or time.time() > self._deadline:
                break

            # Diverse starting points
            if s == 0:
                x0 = np.zeros(dim)
            elif s == 1:
                x0 = np.full(dim, 3.0)
            elif s == 2:
                x0 = np.full(dim, -3.0)
            elif s == 3:
                x0 = rng.uniform(-4, 4, dim)
            else:
                x0 = rng.uniform(-5, 5, dim)

            fx0 = f(x0)
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

            # Simple CMA-ES-like local search around x0
            sigma = 1.0
            x = x0.copy()
            fx = fx0
            # Use a simple evolution strategy with population size 4
            pop_size = 4
            max_iter = per_start_budget
            step = 0
            while step < max_iter and calls[0] < budget:
                if time.time() > self._deadline:
                    break
                # Generate offspring
                for _ in range(pop_size):
                    if calls[0] >= budget or time.time() > self._deadline:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = f(y)
                    if fy < fx:
                        x = y.copy()
                        fx = fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        sigma *= 1.1
                    else:
                        sigma *= 0.95
                    step += 1
                if sigma < 1e-6:
                    break

        # Phase 2: L-BFGS-B polish from best point and a few other good points
        # Collect candidate starting points for polishing
        polish_starts = [best_x.copy()]
        # Add a few deterministic diverse points
        polish_starts.append(np.zeros(dim))
        polish_starts.append(np.full(dim, 3.0))
        polish_starts.append(np.full(dim, -3.0))
        polish_starts.append(rng.uniform(-4, 4, dim))

        for x_start in polish_starts:
            if calls[0] >= budget or time.time() > self._deadline:
                break
            try:
                res = minimize(
                    f,
                    x_start,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'maxfun': 50}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
