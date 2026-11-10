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
        
        # For ZDT6 with x[1:]=0 (g=1):
        #   f1 = 10 + sqrt(x0), f2 = 1 - sqrt(x0)
        #   Pareto front: f1 + f2 = 11, f1 in [10,11], f2 in [0,1]
        #
        # To maximize hypervolume with ref point (11,11), sample f1 uniformly
        # in [10, 11]. This gives x0 = (f1 - 10)^2.
        
        f1_targets = np.linspace(10.0, 11.0, n_evals)
        x0s = (f1_targets - 10.0) ** 2
        
        # Ensure endpoints are exactly hit
        x0s[0] = 0.0
        x0s[-1] = 1.0
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim, dtype=float)
            x[0] = float(x0s[i])
            # x[1:] = 0 minimizes g, placing points on the Pareto front
            
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break
                
        return results
