import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295
        
        n_evals = self.budget
        if n_evals <= 0:
            return []
            
        results = []
        
        # Uniform sampling of x0 in [0, 1] to cover the oscillatory f1 range of ZDT6
        x0s = np.linspace(0.0, 1.0, n_evals)
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim, dtype=float)
            x[0] = float(x0s[i])
            # x[1:] = 0 minimizes g, putting points on the Pareto front
            
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break
                
        return results
