import numpy as np
import time
from scipy.optimize import minimize

class BudgetExceeded(Exception):
    pass

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 320.0 # Safety margin for 330s limit
        
        eval_count = 0
        best_x = None
        best_f = np.inf
        
        rng = np.random.default_rng(42)
        
        # Initial random point
        x0 = rng.uniform(-5, 5, self.dim)
        
        def eval_func(x):
            nonlocal eval_count, best_x, best_f
            eval_count += 1
            if eval_count > self.budget:
                raise BudgetExceeded()
            if time.time() - start_time > time_limit:
                raise BudgetExceeded()
            
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def callback(xk):
            nonlocal eval_count
            if eval_count > self.budget:
                raise BudgetExceeded()
            if time.time() - start_time > time_limit:
                raise BudgetExceeded()

        # Multi-start loop
        # We don't know how many calls each minimize takes, so we loop until budget is exhausted
        # Powell can be fast, so we might get many restarts.
        # Nelder-Mead might be slower per call but more robust?
        # Let's try Powell first.
        
        while True:
            # Generate a new starting point
            # If we have a best_x, we could perturb it, but for global search, uniform is safer
            # Let's use uniform for the first few, then maybe perturb best?
            # For simplicity, uniform.
            x_start = rng.uniform(-5, 5, self.dim)
            
            try:
                res = minimize(
                    eval_func, 
                    x_start, 
                    method='Powell', 
                    options={'maxiter': 50, 'xtol': 1e-4, 'ftol': 1e-4},
                    callback=callback
                )
                # If it finishes naturally, update best if it found something better
                # (eval_func already updates best)
            except BudgetExceeded:
                break
            except Exception:
                # Catch other errors to avoid crashing the whole run
                # If an error occurs, we might still have budget left, but let's be safe
                # and break if we are close to limit or just continue?
                # If minimize crashes, it might have consumed some budget.
                # Let's check budget.
                if eval_count > self.budget * 0.95:
                    break
                continue

            # Check if we should continue
            if eval_count > self.budget:
                break
            if time.time() - start_time > time_limit:
                break
                
            # If we finished a minimize, we can start another
            # To avoid infinite loop if minimize returns immediately (e.g. 1 call), 
            # we rely on the budget check.
            
        return best_x, best_f
