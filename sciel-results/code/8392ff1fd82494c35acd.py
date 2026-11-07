import numpy as np
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
        self.deadline = time.time() + 285  # 15s margin before 300s kill
        self.calls = 0
        self.best_x = np.zeros(self.dim)
        self.best_f = np.inf

        def wrapped_func(x):
            if self.calls >= self.budget:
                return np.inf
            if time.time() > self.deadline:
                return np.inf
            x = np.clip(x, -5, 5)
            self.calls += 1
            f = float(func(x))
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        dim = self.dim
        budget = self.budget

        # Strategy: Multi-start Nelder-Mead simplex optimization
        # Nelder-Mead is a derivative-free direct search method that uses
        # geometric operations (reflection, expansion, contraction) on a simplex.
        # It's well-suited for smooth, low-dimensional problems like Griewank-Rosenbrock.

        # Generate diverse starting points
        np.random.seed(42)
        starts = []
        starts.append(np.zeros(dim))
        for _ in range(8):
            starts.append(np.random.uniform(-4, 4, dim))
        for _ in range(2):
            starts.append(np.random.uniform(-5, -4, dim))
            starts.append(np.random.uniform(4, 5, dim))

        num_starts = len(starts)
        calls_per_start = max(20, (budget - 10) // num_starts)

        from scipy.optimize import minimize

        for i in range(num_starts):
            if self.calls >= budget - 5:
                break
            if time.time() > self.deadline - 5:
                break

            x0 = starts[i].copy()
            f0 = wrapped_func(x0)

            if not np.isfinite(f0):
                continue

            # Use Nelder-Mead for local optimization from this start
            if self.calls < budget - 10 and time.time() < self.deadline - 5:
                try:
                    remaining = min(calls_per_start, budget - self.calls - 5)
                    if remaining > 15:
                        maxiter = max(1, remaining // 6)
                        res = minimize(wrapped_func, x0, method='Nelder-Mead',
                                       options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-12,
                                                'initial_simplex': None})
                        if np.isfinite(res.fun) and res.fun < self.best_f:
                            self.best_f = res.fun
                            self.best_x = np.clip(res.x, -5, 5).copy()
                except Exception:
                    pass

        # Final polish with a tighter Nelder-Mead around the best point
        if self.calls < budget - 3 and time.time() < self.deadline - 3:
            try:
                remaining = min(40, budget - self.calls - 2)
                if remaining > 10:
                    maxiter = max(1, remaining // 5)
                    res = minimize(wrapped_func, self.best_x, method='Nelder-Mead',
                                   options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-14})
                    if np.isfinite(res.fun) and res.fun < self.best_f:
                        self.best_f = res.fun
                        self.best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass

        return self.best_x, self.best_f
