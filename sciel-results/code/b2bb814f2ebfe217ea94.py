import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = float('inf')
        self.call_count = 0
        self.start_time = None
        self.deadline = 300.0 # Safety margin below 330s

    def __call__(self, func):
        self.start_time = time.time()
        
        # Wrapper to track calls and enforce budget/deadline
        def wrapped_func(x):
            if self.call_count >= self.budget:
                return self.best_f
            if time.time() > self.deadline:
                return self.best_f
            
            x = np.asarray(x)
            # Clip to box [-5, 5] as per problem description, though L-BFGS-B handles bounds
            x_clipped = np.clip(x, -5, 5)
            
            try:
                f_val = func(x_clipped)
            except Exception:
                return 1e10 # Return large value on error to guide optimizer away? Or just fail?
                # Problem says exception later keeps calls made. But we want to avoid crashing the optimizer.
                # If func throws, it's likely out of domain or similar. Clipping should prevent domain errors.
                # If it still throws, we return a bad value.
            
            self.call_count += 1
            if f_val < self.best_f:
                self.best_f = f_val
                self.best_x = x_clipped.copy()
            
            return f_val

        # 1. Global Sampling Phase
        # Generate diverse starting points
        rng = np.random.default_rng(42)
        candidates = []
        
        # Center
        candidates.append(np.zeros(self.dim))
        
        # Vertices (2^5 = 32, maybe too many? Let's take a subset or all if budget allows)
        # 500 budget is plenty. 32 vertices + 20 random = 52 calls.
        # Let's use 10 random points to be safe and fast.
        candidates.append(rng.uniform(-5, 5, self.dim))
        for _ in range(9):
            candidates.append(rng.uniform(-5, 5, self.dim))
            
        # Evaluate candidates
        for x_cand in candidates:
            if self.call_count >= self.budget or time.time() > self.deadline:
                break
            f_cand = wrapped_func(x_cand)
        
        # If we didn't find any valid point (shouldn't happen), fallback to center
        if self.best_x is None:
            self.best_x = np.zeros(self.dim)
            self.best_f = wrapped_func(self.best_x)

        # 2. Local Optimization Phase using L-BFGS-B
        # Use the best point found so far as the initial guess
        x0 = self.best_x
        bounds = [(-5.0, 5.0)] * self.dim
        
        # L-BFGS-B is good for box constraints.
        # maxiter: limit iterations to ensure we don't spend too much time/calls
        # maxfun: limit function evaluations
        remaining_budget = self.budget - self.call_count
        if remaining_budget > 10:
            try:
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 100, 'maxfun': remaining_budget, 'gtol': 1e-8}
                )
                # Update best if res is better
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = res.x
            except Exception:
                pass # Keep previous best if optimizer fails

        return self.best_x, self.best_f
