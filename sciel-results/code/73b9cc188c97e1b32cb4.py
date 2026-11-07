import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.call_count = 0
        self.start_time = None
        self.deadline = 290.0  # 10s margin from 300s

    def __call__(self, func):
        self.call_count = 0
        self.best_f = np.inf
        self.best_x = np.zeros(self.dim)
        self.start_time = time.time()

        def wrapped_func(x):
            self.call_count += 1
            
            # Deadline check
            if time.time() - self.start_time > self.deadline:
                raise Exception("Time Exceeded")
                
            # Budget check
            if self.call_count > self.budget:
                raise Exception("Budget Exceeded")

            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            
            if f_val < self.best_f:
                self.best_f = f_val
                self.best_x = x_clipped.copy()
                
            return f_val

        # Initialize at 0
        x0 = np.zeros(self.dim)
        
        try:
            # L-BFGS-B with forward differences
            # maxiter set high, but we stop via exception
            res = minimize(wrapped_func, x0, method='L-BFGS-B', jac='forward', 
                           options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 50})
        except Exception:
            pass

        # If we have budget left, we could run a second phase?
        # But L-BFGS-B should have used most of it.
        # If it finished early (converged), we are done.
        
        return self.best_x, self.best_f
