"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-Start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.best_f = float('inf')
        self.best_x = None
        self.func = None

    def _wrapper(self, x):
        if self.call_count >= self.budget:
            raise OverflowError("Budget exceeded")
        self.call_count += 1
        f = self.func(x)
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def __call__(self, func):
        self.func = func
        self.call_count = 0
        self.best_f = float('inf')
        self.best_x = None
        
        rng = np.random.default_rng(42)
        
        # Multi-start Nelder-Mead
        n_starts = 5
        starts = [np.zeros(self.dim)]
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, self.dim))
            
        for start in starts:
            # Stop if we are close to budget limit to avoid overhead
            if self.call_count >= self.budget - 10:
                break
                
            try:
                # maxiter is a heuristic for stopping, but our hard stop is in _wrapper
                res = minimize(self._wrapper, start, method='Nelder-Mead', 
                               options={'maxiter': self.budget - self.call_count, 'xatol': 1e-5, 'fatol': 1e-5})
            except OverflowError:
                break
            except Exception:
                break
                
        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
