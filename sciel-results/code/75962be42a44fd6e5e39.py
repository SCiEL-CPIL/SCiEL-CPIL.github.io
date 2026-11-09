import numpy as np
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    """
    Optimizer using Differential Evolution for global search 
    followed by Nelder-Mead for local refinement.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0

    def __call__(self, func):
        self.call_count = 0
        bounds = [(-5, 5)] * self.dim
        
        # Define a wrapper to count evaluations and enforce budget
        def wrapped_func(x):
            if self.call_count >= self.budget:
                return 1e10  # Large penalty if budget exceeded
            self.call_count += 1
            # Ensure input is a numpy array of correct shape
            x = np.asarray(x, dtype=float)
            return float(func(x))

        best_x = np.zeros(self.dim)
        best_f = np.inf

        # Phase 1: Global Search with Differential Evolution
        # Allocate ~60% of the budget to global search to ensure good global coverage
        global_budget = int(self.budget * 0.6)
        # DE evaluates popsize * (maxiter + 1) times approximately
        # Increase popsize for better global search in 5D
        popsize = 20
        maxiter = max(1, global_budget // popsize - 2) # Leave some buffer
        
        try:
            res_global = differential_evolution(
                wrapped_func, 
                bounds, 
                maxiter=maxiter, 
                popsize=popsize,
                init='sobol', # Better space-filling than latinhypercube
                seed=0,
                polish=False  # Disable internal polish to save budget for our specific local step
            )
            if res_global.fun < best_f:
                best_x = res_global.x
                best_f = res_global.fun
        except Exception:
            # Fallback: if DE fails, use a random point or origin
            best_x = np.zeros(self.dim)
            best_f = wrapped_func(best_x)

        # Phase 2: Local Refinement with Nelder-Mead
        remaining_budget = self.budget - self.call_count
        if remaining_budget > 10:
            try:
                # Nelder-Mead is robust and fast for low dimensions
                res_local = minimize(
                    wrapped_func, 
                    best_x, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining_budget,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if res_local.fun < best_f:
                    best_x = res_local.x
                    best_f = res_local.fun
            except Exception:
                pass

        # Ensure we return a valid point even if something went wrong
        if not np.isfinite(best_f):
             best_x = np.zeros(self.dim)
             best_f = wrapped_func(best_x)

        return best_x, best_f
