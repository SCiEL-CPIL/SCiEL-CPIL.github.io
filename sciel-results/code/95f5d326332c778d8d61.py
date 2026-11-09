"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-restart L-BFGS-B optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(self.dim)
        
        def counted_func(x):
            if self.calls >= self.budget:
                raise Exception("Budget exceeded")
            self.calls += 1
            x = np.clip(x, -5, 5)
            fx = func(x)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        def grad(x):
            g = np.zeros(self.dim)
            h = 1e-5
            x0 = x.copy()
            fx0 = counted_func(x0)
            for i in range(self.dim):
                x_plus = x0.copy()
                x_plus[i] += h
                fx_plus = counted_func(x_plus)
                g[i] = (fx_plus - fx0) / h
            return g

        rng = np.random.default_rng(42)
        
        # Define starting points
        starts = [np.zeros(self.dim)]
        for _ in range(19):
            starts.append(rng.uniform(-5, 5, self.dim))
            
        for x0 in starts:
            if self.calls >= self.budget - 5:
                break
                
            try:
                remaining = self.budget - self.calls
                max_iter = max(1, int(remaining / 20))
                
                res = minimize(
                    counted_func, 
                    x0, 
                    jac=grad, 
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': max_iter, 'gtol': 1e-5}
                )
                
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x.copy()
            except Exception:
                pass

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
