"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        # Powell's method is derivative-free and adapts search directions.
        # It is particularly good for smooth functions like Rosenbrock,
        # even when rotated, because it effectively finds the conjugate directions.
        
        # We will use a multi-restart strategy.
        # Powell can be sensitive to the initial simplex, so multiple starts are beneficial.
        # Each Powell run is relatively cheap in terms of function evaluations compared to gradient methods
        # if it converges quickly, but can be expensive if it wanders.
        
        # Budget management:
        # We don't have a direct hook to count calls in scipy's minimize easily without wrapping.
        # We will wrap the function to count calls.
        
        call_count = 0
        def counted_func(x):
            nonlocal call_count
            call_count += 1
            return func(x)
        
        bounds = [(-5, 5)] * dim
        rng = np.random.default_rng(42)
        
        # Strategy:
        # 1. Try a few random starts.
        # 2. Use Powell's method.
        # 3. Set maxiter to limit the cost per run.
        # Powell's method in scipy has 'maxiter' which is the number of iterations of the outer loop.
        # Each outer iteration involves line searches in each direction.
        # For dim=5, one outer iteration might take ~5-10 evaluations if line searches are efficient.
        # Let's allow up to 500 evaluations total.
        # If we do 5 restarts, each gets ~100 evaluations.
        # maxiter=20 for Powell might be too low or too high depending on convergence.
        # Let's use maxiter=50 to be safe, but break early if no improvement.
        
        num_restarts = 5
        
        for i in range(num_restarts):
            if time.time() > deadline:
                break
            if call_count >= budget:
                break
            
            # Random start
            # For Rosenbrock, the minimum is at (1,1,...,1) but here it's shifted/rotated.
            # Uniform random in [-5, 5] is a reasonable global start.
            x0 = rng.uniform(-5, 5, dim)
            
            # Run Powell's method
            try:
                # ftol and xtol control convergence
                # maxiter limits the number of iterations
                res = minimize(counted_func, x0, method='Powell', bounds=bounds, 
                               options={'maxiter': 50, 'ftol': 1e-8, 'xtol': 1e-8})
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass
        
        # If we have budget left, do a final polish with a more precise method?
        # Powell is already quite precise.
        # Maybe try Nelder-Mead on the best result found?
        if call_count < budget - 10 and time.time() < deadline:
            try:
                res = minimize(counted_func, best_x, method='Nelder-Mead', bounds=bounds,
                               options={'maxiter': 100, 'fatol': 1e-10, 'xatol': 1e-10})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
