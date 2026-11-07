import numpy as np
from scipy.optimize import differential_evolution

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = [(-5, 5)] * self.dim
        popsize = 5
        n = self.dim
        
        # Calculate max iterations to stay within budget
        # DE uses popsize*n for initial pop and each generation
        # Total evals = (maxiter + 1) * popsize * n
        # We want Total <= budget
        # maxiter + 1 <= budget / (popsize * n)
        # maxiter <= budget / (popsize * n) - 1
        
        max_iter = int(self.budget / (popsize * n)) - 1
        if max_iter < 1:
            max_iter = 1

        try:
            result = differential_evolution(
                func, 
                bounds, 
                popsize=popsize, 
                maxiter=max_iter, 
                seed=0,
                tol=1e-8,
                polish=False
            )
            return result.x, result.fun
        except Exception:
            # Fallback to simple random search if DE fails
            rng = np.random.default_rng(0)
            best_x = rng.uniform(-5, 5, n)
            best_f = func(best_x)
            for _ in range(self.budget - 1):
                x = rng.uniform(-5, 5, n)
                f = func(x)
                if f < best_f:
                    best_f = f
                    best_x = x
            return best_x, best_f
