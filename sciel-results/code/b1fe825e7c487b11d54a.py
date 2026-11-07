# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start scout + Powell polish for Rosenbrock-style f8."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 300.0  # safety margin below 330s kill

        # Track best
        rng = np.random.default_rng(0)

        def time_ok():
            return time.time() < deadline - 5.0  # 5s margin

        # --- Phase 1: Global scout with a small population of random starts ---
        # Rosenbrock has a narrow valley; we need to get close to the basin first.
        # Use a structured set of starting points including origin and random.
        starts = [np.zeros(self.dim)]
        # Add structured starts around the origin and random ones
        n_scout = 20
        for _ in range(n_scout):
            starts.append(rng.uniform(-5, 5, self.dim))
        # A few starts biased toward origin (Rosenbrock min is near origin in shifted coords)
        for r in [1.0, 2.0, 3.0, 4.0]:
            for _ in range(3):
                starts.append(rng.uniform(-r, r, self.dim))

        # Evaluate all starts, keep the best
        best_x = None
        best_f = np.inf
        evaluated = 0
        for s in starts:
            if not time_ok():
                break
            s = np.clip(s, -5, 5)
            f = func(s)
            evaluated += 1
            if f < best_f:
                best_f = f
                best_x = s.copy()

        # --- Phase 2: Powell polish from the best start ---
        # Powell's method is derivative-free and excels on Rosenbrock valleys.
        def wrapped(x):
            x = np.clip(x, -5, 5)
            return func(x)

        try:
            res = minimize(
                wrapped,
                best_x,
                method='Powell',
                options={
                    'maxiter': 200,
                    'maxfev': 200,
                    'ftol': 1e-12,
                    'xtol': 1e-12,
                },
            )
            if res.fun < best_f:
                best_f = res.fun
                best_x = np.clip(res.x, -5, 5)
        except Exception:
            pass

        # --- Phase 3: If time and budget remain, do additional restarts + Powell ---
        remaining = self.budget - evaluated
        if remaining > 0 and time_ok():
            n_extra = min(10, remaining // 50)  # each Powell uses ~50-100 evals
            for i in range(n_extra):
                if not time_ok():
                    break
                s = rng.uniform(-5, 5, self.dim)
                s = np.clip(s, -5, 5)
                f = func(s)
                evaluated += 1
                if f < best_f:
                    best_f = f
                    best_x = s.copy()
                # Quick Powell from this start
                try:
                    res = minimize(
                        wrapped,
                        best_x,
                        method='Powell',
                        options={
                            'maxiter': 100,
                            'maxfev': 100,
                            'ftol': 1e-12,
                            'xtol': 1e-12,
                        },
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
