import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = None

    def __call__(self, func):
        self.start_time = time.time()
        self.deadline = self.start_time + 300.0 * 0.95
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf

        rng = np.random.default_rng(123)

        def wrapper(x):
            self.calls += 1
            fx = func(x)
            if np.isnan(fx) or np.isinf(fx):
                return 1e10
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # Restart strategy
        # (budget, mode, perturbation_std)
        restarts = [
            (200, 'random', 0.0),
            (150, 'perturb', 2.0),
            (100, 'random', 0.0),
            (50, 'perturb', 0.5)
        ]

        for sub_budget, mode, std in restarts:
            if self.calls >= self.budget:
                break
            if time.time() > self.deadline:
                break

            if mode == 'random':
                x_start = rng.uniform(-5, 5, self.dim)
            else:
                if self.best_x is None:
                    x_start = rng.uniform(-5, 5, self.dim)
                else:
                    x_start = self.best_x + rng.normal(0, std, self.dim)
                    x_start = np.clip(x_start, -5, 5)

            start_calls = self.calls

            def callback(xk):
                if self.calls - start_calls >= sub_budget:
                    return True
                if time.time() > self.deadline:
                    return True
                return False

            try:
                minimize(wrapper, x_start, method='Powell', 
                         options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-12},
                         callback=callback)
            except Exception:
                pass

        return self.best_x, self.best_f
