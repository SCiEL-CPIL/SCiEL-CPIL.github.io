import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        best_x = None
        best_f = float('inf')
        
        # Use multiple random restarts to handle multimodality and rotation.
        # Nelder-Mead is robust to rotation compared to coordinate descent.
        n_restarts = 5
        budget_per_restart = budget // n_restarts
        
        # We will manage the total evaluation count manually to ensure we don't exceed 500.
        # scipy's minimize doesn't easily let us interrupt mid-run based on a global counter without wrapping.
        # We can wrap func to count calls.
        
        eval_count = 0
        def wrapped_func(x):
            nonlocal eval_count
            if eval_count >= budget:
                return best_f # Return current best to stop search? Or raise? 
            # Actually, if we exceed, we should stop.
            # But scipy might call it multiple times in a loop.
            # Let's just count.
            eval_count += 1
            return func(x)

        # Initial random points
        # Uniform in [-5, 5]
        rng = np.random.default_rng(42)
        
        for r in range(n_restarts):
            if eval_count >= budget or time.time() > deadline:
                break
                
            # Generate a random start point
            x0 = rng.uniform(-5, 5, dim)
            
            # Run Nelder-Mead
            # maxiter is a rough guide, but we control via eval_count?
            # It's better to limit maxiter to ensure we don't burn all budget in one bad start.
            # 5D Nelder-Mead simplex has 6 points. Each iteration is ~1-2 evals.
            # 100 evals ~ 50 iterations.
            
            try:
                res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                               options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
                    
            except Exception:
                pass

            # If we used a lot of budget, break
            if eval_count >= budget:
                break

        # If no valid best_x found (shouldn't happen), return random
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            eval_count += 1

        return best_x, best_f
