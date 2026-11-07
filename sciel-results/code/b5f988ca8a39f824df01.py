import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """Nelder-Mead optimizer with random restarts for black-box continuous functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        calls_made = [0]
        best_x = None
        best_f = float('inf')
        
        def wrapped_func(x):
            if calls_made[0] >= self.budget:
                return 1e30
            calls_made[0] += 1
            val = func(x)
            return val

        rng = np.random.default_rng(42)
        
        # We will perform multiple restarts.
        # Each restart uses a portion of the budget.
        # Nelder-Mead typically uses dim+1 points initially, and then 1 per iteration.
        # Let's allow up to budget//5 calls per restart, or until convergence.
        
        # Strategy: 
        # 1. Initial random scan to find a good starting point? 
        #    With dim=5, a grid is too expensive. Random sampling is better.
        # 2. Run Nelder-Mead from several random starts.
        
        # Let's do 5 restarts. Each gets 100 calls.
        # If a restart finishes early, we can use the remaining budget for another.
        
        max_restarts = 10
        calls_per_restart = max(10, self.budget // max_restarts)
        
        for i in range(max_restarts):
            if calls_made[0] >= self.budget:
                break
            
            x0 = rng.uniform(-5, 5, self.dim)
            
            # Nelder-Mead options
            # maxiter is not strictly function count, but it correlates.
            # We use maxfun to limit function calls.
            # However, minimize's maxfun stops the optimization.
            
            # We need to be careful: minimize might not respect maxfun exactly if it's in the middle of an iteration?
            # Actually, scipy's minimize with method='Nelder-Mead' respects maxfun.
            
            res = minimize(
                wrapped_func,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000, # High enough to not stop on iterations
                    'maxfun': calls_per_restart, 
                    'xatol': 1e-6,
                    'fatol': 1e-6,
                    'adaptive': True
                }
            )
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        
        return best_x, best_f
