import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func = None
        self.best_x = None
        self.best_f = np.inf
        self.call_count = 0
        self.start_time = None

    def __call__(self, func):
        self.func = func
        self.start_time = time.time()
        self.best_x = np.zeros(self.dim)
        self.best_f = np.inf
        self.call_count = 0

        # Generate 10 diverse initial points
        rng = np.random.default_rng(123)
        # Use a mix of random and some structured points?
        # Let's just use random.
        n_restarts = 10
        initial_points = rng.uniform(-5, 5, (n_restarts, self.dim))
        
        # Budget per restart
        budget_per_restart = self.budget // n_restarts
        
        for x0 in initial_points:
            # Time check
            if time.time() - self.start_time > 250:
                break
            
            # Budget check
            if self.call_count >= self.budget:
                break
            
            # Run L-BFGS-B
            # maxfun is the maximum number of function evaluations
            # We give it the remaining budget or the per-restart budget, whichever is smaller
            # But we want to use the full budget if possible.
            # If one restart finishes early, the next one can use more.
            # So we pass the remaining budget as maxfun.
            remaining_budget = self.budget - self.call_count
            if remaining_budget <= 0:
                break
                
            res = minimize(
                self._wrapped_func, 
                x0, 
                method='L-BFGS-B', 
                bounds=[(-5, 5)] * self.dim,
                options={'maxfun': remaining_budget, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            # The best is already updated in _wrapped_func
        
        return self.best_x, self.best_f

    def _wrapped_func(self, x):
        x = np.clip(x, -5, 5)
        self.call_count += 1
        f = self.func(x)
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f
