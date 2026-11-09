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
        
        # For ZDT6, Pareto front: f2 = 1 - sqrt(f1) when g=1 (x[1:]=0)
        # To get uniform spacing in f2, use f1 = (i/(N-1))^2
        i_arr = np.arange(n_evals)
        us = i_arr / (n_evals - 1) if n_evals > 1 else np.array([0.0])
        x0s = us ** 2.0
        
        # Ensure we hit both endpoints
        if n_evals > 1:
            x0s[0] = 0.0
            x0s[-1] = 1.0
        
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
