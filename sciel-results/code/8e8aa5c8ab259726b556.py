import numpy as np
import time
from scipy.optimize import powell

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        calls = 0
        best_x = np.zeros(self.dim)
        best_f = float('inf')

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return best_f # Return best known to stop optimization
            if time.time() - start_time > 280:
                return best_f
                
            calls += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initial guess: 0
        x0 = np.zeros(self.dim)
        
        # Try to run Powell
        try:
            res = powell(wrapped_func, x0, bounds=[(-5, 5)] * self.dim, maxiter=100, maxfev=self.budget)
        except Exception:
            pass

        return best_x, best_f
