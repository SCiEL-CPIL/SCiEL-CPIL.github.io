import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, func):
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf

        def wrapped_func(x):
            self.call_count += 1
            # Clip to ensure we are in the box for the actual function evaluation
            # Note: COBYLA will respect constraints, but clipping is a safety net
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x_clipped.copy()
            return fx

        # Define box constraints for COBYLA
        # COBYLA constraints are g(x) >= 0
        constraints = []
        for i in range(self.dim):
            # x_i >= -5  =>  x_i + 5 >= 0
            constraints.append(lambda x, i=i: x[i] + 5.0)
            # x_i <= 5   =>  5 - x_i >= 0
            constraints.append(lambda x, i=i: 5.0 - x[i])

        x0 = np.zeros(self.dim)
        half_budget = self.budget // 2
        
        # Phase 1
        try:
            res1 = minimize(wrapped_func, x0, method='COBYLA', 
                            constraints=constraints,
                            options={'maxiter': 100, 'maxfev': half_budget})
        except Exception:
            res1 = None

        # Phase 2: Restart from best found so far
        if res1 is not None and self.best_x is not None:
            x0_phase2 = self.best_x
        else:
            x0_phase2 = np.random.uniform(-5, 5, self.dim)
            
        remaining_budget = self.budget - self.call_count
        if remaining_budget > 0:
            try:
                res2 = minimize(wrapped_func, x0_phase2, method='COBYLA', 
                                constraints=constraints,
                                options={'maxiter': 100, 'maxfev': remaining_budget})
            except Exception:
                res2 = None

        return self.best_x, self.best_f
