import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, func):
        self.start_time = time.time()
        self.calls = 0
        
        # Wrap function to count calls and enforce deadline
        def wrapped_func(x):
            if self.calls >= self.budget:
                # Return a large value to stop optimization or raise
                # Raising might be safer to break the loop immediately if SciPy doesn't handle it well
                # But let's just return a large value, SciPy might continue a bit but we will check after
                pass
            self.calls += 1
            # Check time
            if time.time() - self.start_time > 280: # 20s margin
                # We can't easily stop SciPy from inside without exception
                # Let's rely on maxiter and maxfev
                pass
            return func(x)

        # Initial Global Search
        # Use 20% of budget for global search
        global_budget = max(10, self.budget // 5)
        
        best_x = None
        best_f = np.inf
        
        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(global_budget):
            if self.calls >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = wrapped_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x
        
        self.best_x = best_x
        self.best_f = best_f

        if self.calls >= self.budget:
            return self.best_x, self.best_f

        # Local Search with Powell's Method
        # Powell is good for ill-conditioned problems
        # maxfev is the remaining budget
        remaining_budget = self.budget - self.calls
        
        try:
            res = minimize(
                wrapped_func, 
                best_x, 
                method='Powell', 
                options={'maxfev': remaining_budget, 'xtol': 1e-10, 'ftol': 1e-10}
            )
            if res.fun < self.best_f:
                self.best_x = res.x
                self.best_f = res.fun
        except Exception:
            pass

        return self.best_x, self.best_f
