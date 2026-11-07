"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer: L-BFGS-B multi-start + Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = None

    def __call__(self, func):
        self.deadline = time.time() + 280
        self.calls = 0
        self.best_x = np.zeros(self.dim)
        self.best_f = np.inf

        def wrapped_func(x):
            if self.calls >= self.budget:
                return np.inf
            if time.time() > self.deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            self.calls += 1
            f = float(func(x))
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        # Phase 1: Initial sampling to get a sense of the landscape
        n_initial = 25
        rng = np.random.default_rng(42)
        initial_points = rng.uniform(-4, 4, (n_initial, self.dim))
        initial_points[0] = np.zeros(self.dim)
        initial_points[1] = rng.uniform(-5, 5, self.dim)

        for x0 in initial_points:
            if self.calls >= self.budget or time.time() > self.deadline - 30:
                break
            wrapped_func(x0)

        # Phase 2: L-BFGS-B from multiple diverse starting points
        # Use the best found point and several random points as seeds
        lbfgs_starts = [
            self.best_x.copy(),
            rng.uniform(-3, 3, self.dim),
            rng.uniform(-4, 4, self.dim),
            np.zeros(self.dim),
            rng.uniform(-2, 2, self.dim),
        ]

        for x0 in lbfgs_starts:
            if self.calls >= self.budget or time.time() > self.deadline - 40:
                break
            try:
                res = minimize(wrapped_func, x0, method='L-BFGS-B',
                               bounds=[(-5, 5)] * self.dim,
                               options={'maxiter': 300, 'maxfun': 500, 'gtol': 1e-8})
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Nelder-Mead polish on the best result
        if self.calls < self.budget and time.time() < self.deadline - 15:
            try:
                # Create a small initial simplex around best_x
                simplex = [self.best_x.copy()]
                for i in range(self.dim):
                    v = self.best_x.copy()
                    step = 0.05
                    v[i] += step
                    simplex.append(v)
                
                res = minimize(wrapped_func, self.best_x, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-12,
                                        'initial_simplex': np.array(simplex),
                                        'adaptive': True})
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x.copy()
            except Exception:
                pass

        # Phase 4: Final local random search with decreasing step size
        if self.calls < self.budget and time.time() < self.deadline - 5:
            remaining = min(self.budget - self.calls, 150)
            for i in range(remaining):
                if time.time() > self.deadline - 5:
                    break
                step = 0.2 * (0.7 ** (i // 20))
                x_try = self.best_x + rng.normal(0, step, self.dim)
                x_try = np.clip(x_try, -5, 5)
                f_try = wrapped_func(x_try)
                # wrapped_func already updates best if improved

        # Ensure we have a valid result
        if self.best_x is None:
            self.best_x = np.zeros(self.dim)
            self.best_f = func(self.best_x)

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
