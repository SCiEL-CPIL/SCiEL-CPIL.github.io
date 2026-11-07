import time
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None

    def __call__(self, func):
        self.calls = 0
        self.start_time = time.time()
        limit = self.budget
        time_limit = 290.0 # 10s margin
        
        def counted_func(x):
            if time.time() - self.start_time > time_limit:
                raise TimeoutError("Time limit exceeded")
            self.calls += 1
            x = np.asarray(x, dtype=float).flatten()
            if x.shape[0] != self.dim:
                x = x[:self.dim]
            return func(x)

        rng = np.random.default_rng(42)
        
        # Phase 1: Initial Exploration
        n_init = min(20, self.budget // 5)
        
        best_x = None
        best_f = np.inf
        
        for _ in range(n_init):
            try:
                x = rng.uniform(-5, 5, self.dim)
                f = counted_func(x)
                if f < best_f:
                    best_f = f
                    best_x = x
            except TimeoutError:
                break
        
        if best_x is None:
            # Should not happen, but handle it
            return np.zeros(self.dim), np.inf

        # Phase 2: Nelder-Mead Optimization
        remaining = self.budget - self.calls
        
        if remaining > 10:
            try:
                result = minimize(
                    counted_func, 
                    best_x, 
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'maxfev': remaining, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x
            except TimeoutError:
                pass
            except Exception:
                # Fallback
                self._simple_es(counted_func, best_x, best_f, remaining)
        
        return best_x, best_f

    def _simple_es(self, func, x, fx, budget):
        rng = np.random.default_rng(123)
        sigma = 2.0
        for _ in range(budget):
            if self.calls >= self.budget:
                break
            try:
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9
            except TimeoutError:
                break
        return x, fx
