import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, use 280s to be safe
        deadline = start_time + 280.0
        self.calls_made = 0
        self.best_f = float('inf')
        self.best_x = None

        def wrapped_func(x):
            # Check budget and time
            if self.calls_made >= self.budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            
            self.calls_made += 1
            try:
                # Ensure x is a numpy array and clipped
                x_arr = np.asarray(x, dtype=float)
                # The problem statement says func clips, but we can also clip here 
                # to ensure we are within bounds for any internal logic if needed, 
                # though func handles it.
                val = float(func(x_arr))
                if val < self.best_f:
                    self.best_f = val
                    self.best_x = x_arr.copy()
                return val
            except Exception:
                return float('inf')

        # 1. Global Exploration
        # Use a portion of the budget for random restarts
        # For dim=5, 10-20 starts is reasonable
        n_starts = 15
        rng = np.random.default_rng(123)
        
        for i in range(n_starts):
            if self.calls_made >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            wrapped_func(x)

        # 2. Local Refinement
        # Pick the best starting point found
        if self.best_x is not None:
            if self.calls_made < self.budget and time.time() < deadline:
                # Use Nelder-Mead for local search
                # maxiter is number of iterations, each uses dim+1 evals
                # We have budget - calls_made left
                # Let scipy run, but it will stop when wrapped_func returns inf
                try:
                    res = minimize(wrapped_func, self.best_x, method='Nelder-Mead', 
                                   options={'adaptive': True, 'xatol': 1e-8, 'fatol': 1e-8})
                    # The result is already tracked in self.best_f/self.best_x
                except Exception:
                    pass

        # 3. If we still have budget, try a second round of refinement on the best point
        # or a few more random starts if the first batch was poor
        if self.calls_made < self.budget * 0.8 and time.time() < deadline:
             # Optional: One more Nelder-Mead run if it stopped early due to convergence
             # But usually it runs until budget or convergence.
             pass

        return self.best_x, self.best_f
