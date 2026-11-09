import numpy as np
import time

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320
        
        if HAS_CMA:
            x0 = [0.0] * self.dim
            sigma0 = 1.0
            popsize = 10
            # We have 500 budget.
            # 50 iterations * 10 popsize = 500 evals.
            # Set maxiter to 50.
            options = {
                'maxiter': 50,
                'verbose': -9,
                'seed': 0,
                'bounds': [-5.0] * self.dim + [5.0] * self.dim,
                'CMA_adapt_sigma': 'default'
            }
            
            eval_count = 0
            def wrapper(x):
                nonlocal eval_count
                eval_count += 1
                if time.time() > deadline - 5:
                    # Return a large value to stop optimization quickly?
                    # Or just raise an exception?
                    # Better to return a value and let it finish, 
                    # but we need to stop.
                    # cma doesn't easily support mid-run abort.
                    # So we rely on maxiter.
                    pass
                return func(x)

            # Run CMA-ES
            # fmin returns [x, f, es, log]
            try:
                res = cma.fmin(wrapper, x0, sigma0, options=options)
                return res[0], res[1]
            except Exception:
                # Fallback to simple ES if cma fails
                pass

        # Fallback: (1+1)-ES with restarts
        rng = np.random.default_rng(0)
        best_x = None
        best_f = float('inf')
        
        remaining = self.budget
        restarts = 0
        max_restarts = 5
        
        while remaining > 0 and restarts <= max_restarts:
            # Start new run
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            remaining -= 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            sigma = 1.0
            no_improve = 0
            limit = 100 # Max evals per restart segment
            
            while remaining > 0 and no_improve < limit:
                if time.time() > deadline - 5:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                remaining -= 1
                no_improve += 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.5
                    no_improve = 0
                else:
                    sigma *= 0.87
                    
            restarts += 1
            
        return best_x, best_f
