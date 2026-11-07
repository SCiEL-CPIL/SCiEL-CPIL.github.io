import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.func = None
        self.best_x = None
        self.best_f = float('inf')
        self.start_time = None
        self.deadline = None

    def _evaluate(self, x):
        if self.calls >= self.budget:
            return 1e10
        if time.time() > self.deadline:
            return 1e10
        self.calls += 1
        val = self.func(x)
        if val < self.best_f:
            self.best_f = val
            self.best_x = np.array(x)
        return val

    def __call__(self, func):
        self.func = func
        self.calls = 0
        self.best_f = float('inf')
        self.best_x = None
        self.start_time = time.time()
        self.deadline = self.start_time + 280.0

        d = self.dim
        rng = np.random.default_rng(42)

        def wrapper_func(x):
            if self.calls >= self.budget or time.time() > self.deadline - 2:
                return 1e10
            return self._evaluate(x)

        # Phase 1: Initial random sampling to find good starting points
        n_init = 8
        init_points = []
        for _ in range(n_init):
            if self.calls >= self.budget - 50:
                break
            x = rng.uniform(-5, 5, size=d)
            f = wrapper_func(x)
            if f < 1e9:
                init_points.append((f, x))

        if not init_points:
            x0 = np.zeros(d)
            f0 = wrapper_func(x0)
            init_points = [(f0, x0)]

        # Sort by function value
        init_points.sort(key=lambda item: item[0])
        best_start = init_points[0][1]

        # Phase 2: L-BFGS-B from best initial point
        try:
            res = minimize(
                wrapper_func,
                best_start,
                method='L-BFGS-B',
                bounds=[(-5, 5)] * d,
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        except Exception:
            pass

        # Phase 3: Try additional random starts if budget allows
        for _ in range(3):
            if self.calls >= self.budget - 50 or time.time() > self.deadline - 20:
                break
            x_new = rng.uniform(-4, 4, size=d)
            f_new = wrapper_func(x_new)
            if f_new < 1e9 and f_new < self.best_f:
                try:
                    minimize(
                        wrapper_func,
                        x_new,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * d,
                        options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                except Exception:
                    pass

        # Phase 4: Nelder-Mead polish for final convergence
        if self.calls < self.budget - 10 and time.time() < self.deadline - 10 and self.best_x is not None:
            try:
                minimize(
                    wrapper_func,
                    self.best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
            except Exception:
                pass

        if self.best_x is None:
            self.best_x = np.zeros(d)
            self.best_f = 0.0

        return self.best_x, self.best_f
