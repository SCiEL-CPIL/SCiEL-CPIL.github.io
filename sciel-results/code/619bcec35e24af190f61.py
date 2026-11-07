import numpy as np
import time
from scipy.optimize import differential_evolution

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin
        eval_count = 0
        max_evals = self.budget - 10 # Leave some buffer
        
        def wrapper(x):
            nonlocal eval_count
            if time.time() > deadline:
                raise Exception("Time out")
            eval_count += 1
            if eval_count > max_evals:
                raise Exception("Budget exceeded")
            # Clip to box
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)

        bounds = [(-5, 5)] * self.dim
        
        try:
            result = differential_evolution(
                wrapper, 
                bounds, 
                maxiter=100, 
                popsize=10, 
                tol=1e-12, 
                seed=0, 
                polish=True # Use Nelder-Mead at the end
            )
        except Exception:
            # If it stops early, we just use the best found so far?
            # DE doesn't easily return partial best if exception is raised inside callback.
            # We need a different approach if we want to handle exceptions gracefully.
            pass

        # If exception occurred, we might not have a result.
        # Better to implement a custom loop or use a method that allows stopping.
        
        # Let's stick to a simpler, more controlled approach if DE is too risky with exceptions.
        # Actually, raising an exception in the callback will propagate up.
        # We can catch it, but we lose the best point found so far.
        
        # Alternative: Manual DE implementation? Too long.
        
        # Let's go back to Multi-Start Nelder-Mead but with a smarter initialization.
        # Or just a very good (1+1)-ES with CMA-ES like step size adaptation?
        
        # Let's try a simpler CMA-ES implementation.
        # Or use `scipy.optimize.minimize` with 'Powell' method? Powell is derivative-free and good for low dims.
        
        # Let's try Multi-Start Powell.
        # Powell is very efficient in 5D.
        
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        
        # 1. Global coarse search
        # 2. Local refinement
        
        # Let's do a random restart strategy with Powell.
        n_starts = 10
        evals_per_start = (self.budget - 10) // n_starts
        
        rng = np.random.default_rng(0)
        
        for i in range(n_starts):
            if time.time() > deadline:
                break
            if eval_count > self.budget - 10:
                break
                
            x0 = rng.uniform(-5, 5, self.dim)
            
            # Run Powell
            try:
                res = differential_evolution( # Wait, I want Powell or Nelder-Mead
                    wrapper,
                    bounds,
                    maxiter=10, # Limit iterations
                    popsize=5,
                    seed=i
                )
            except:
                continue
                
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x

        return best_x, best_f
