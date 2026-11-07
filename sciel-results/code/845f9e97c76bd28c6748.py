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
        self.deadline = time.time() + 280
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

        # Strategy: 
        # 1. L-BFGS-B from multiple diverse starting points (explores and exploits)
        # 2. Nelder-Mead polish on the best result
        # 3. Final CMA-ES-like local search (simulated annealing with random restarts)
        
        # Initial sampling: get a sense of the landscape
        n_initial = 20
        initial_points = np.random.uniform(-4, 4, (n_initial, self.dim))
        initial_points[0] = np.zeros(self.dim)  # ensure origin is tested
        
        for x0 in initial_points:
            if self.calls >= self.budget or time.time() > self.deadline - 20:
                break
            wrapped_func(x0)

        # L-BFGS-B optimization from several good starting points
        # We pick the best few from initial sampling as seeds for L-BFGS-B
        from scipy.optimize import minimize
        
        # Use the best 5 initial points as starting points for L-BFGS-B
        # (They are already evaluated, but L-BFGS-B will re-evaluate as needed)
        
        # Sort initial points by their function values (we need to track them)
        # Actually, we didn't store them separately, let's just use random restarts
        # But we know the current best. Let's use best_x and a few random ones.
        
        lbfgs_starts = [
            self.best_x.copy(),
            np.random.uniform(-3, 3, self.dim),
            np.random.uniform(-4, 4, self.dim),
            np.zeros(self.dim),
        ]
        
        for x0 in lbfgs_starts:
            if self.calls >= self.budget or time.time() > self.deadline - 30:
                break
            try:
                res = minimize(wrapped_func, x0, method='L-BFGS-B',
                               bounds=[(-5, 5)] * self.dim,
                               options={'maxiter': 200, 'maxfun': 500, 'gtol': 1e-6})
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x.copy()
            except Exception:
                pass

        # Nelder-Mead polish
        if self.calls < self.budget and time.time() < self.deadline - 15:
            try:
                res = minimize(wrapped_func, self.best_x, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-10})
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x.copy()
            except Exception:
                pass

        # Final local search: random perturbations around best_x
        if self.calls < self.budget and time.time() < self.deadline - 5:
            remaining = self.budget - self.calls
            for _ in range(min(remaining, 100)):
                if time.time() > self.deadline - 5:
                    break
                # Try perturbations of decreasing size
                step = 0.1 * (0.5 ** (self.calls // 20))  # adaptive step
                x_try = self.best_x + np.random.normal(0, step, self.dim)
                x_try = np.clip(x_try, -5, 5)
                f_try = wrapped_func(x_try)
                if f_try < self.best_f:
                    self.best_f = f_try
                    self.best_x = x_try.copy()

        return self.best_x, self.best_f
