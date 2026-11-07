import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        calls = 0
        best_x = None
        best_f = np.inf
        time_limit = 290  # Safety margin before 300s limit

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            # Check time occasionally to avoid overhead on every call? 
            # But we need to stop if we exceed budget or time.
            # The contract says >500 calls invalid.
            # We will check budget strictly.
            if calls >= self.budget:
                raise Exception("Budget exceeded")
            
            calls += 1
            
            val = func(x)
            
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float).copy()
                
            return val

        # Multi-Start Strategy
        # We will try a few random starting points and run a local optimizer from each.
        # Powell is good for ill-conditioned functions.
        
        rng = np.random.default_rng(42)
        
        # Determine number of restarts
        # Each Powell run might take ~10-50 evaluations.
        # Let's aim for ~10 restarts.
        num_restarts = 10
        evals_per_restart = int(self.budget / num_restarts)
        
        for i in range(num_restarts):
            # Check time
            if time.time() - start_time > time_limit:
                break
            
            # Generate random start point
            x0 = rng.uniform(-5, 5, self.dim)
            
            # Run Powell
            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='Powell',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 100, 'maxfev': evals_per_restart, 'xtol': 1e-8, 'ftol': 1e-8}
                )
                # Note: Powell might use more evals than maxfev if not careful, 
                # but maxfev should stop it.
                
            except (Exception):
                # If it fails or runs out of budget, continue to next restart
                pass

            if calls >= self.budget:
                break

        # Final polish on the best point found so far
        if best_x is not None and calls < self.budget and time.time() - start_time < time_limit:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='Powell',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 100, 'maxfev': self.budget - calls, 'xtol': 1e-10, 'ftol': 1e-10}
                )
            except (Exception):
                pass

        return best_x, best_f
