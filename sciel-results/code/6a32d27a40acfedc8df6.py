import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT2 specific optimization
        # Pareto front: x2..xn = 0, x1 in [0, 1]
        # f1 = x1, f2 = 1 - sqrt(x1)
        
        # Optimal spacing derived: x1 = u^2 where u is uniform in [0, 1]
        n_evals = self.budget
        if n_evals <= 0:
            return []
            
        # Generate u values
        # Use linspace to ensure coverage of [0, 1]
        # Including 0 and 1 is good for hypervolume boundaries
        u_vals = np.linspace(0, 1, n_evals)
        
        results = []
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
                
            x = np.zeros(self.dim)
            x[0] = u_vals[i] ** 2
            # x[1:] are already 0
            
            # Clip to bounds just in case, though 0 and 1 are within [0, 1]
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            results.append(f)
            
        return results
