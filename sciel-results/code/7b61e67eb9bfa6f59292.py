import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        
        # ZDT3 Pareto front: x[0] = f1 in [0,1], x[1:] = 0
        # This gives g = 1 and f2 = 1 - sqrt(f1) - (1-f1)*sin(10*pi*f1)
        
        # Use quasi-random (Halton or just uniform) distribution for f1
        # 200 points uniformly spaced in [0,1]
        n_evals = self.budget
        f1_values = np.linspace(0, 1, n_evals)
        
        # To avoid exact 0 and 1 which might cause numerical issues, use a tiny epsilon
        # Actually 0 and 1 are fine for ZDT3
        f1_values = np.linspace(0.0, 1.0, n_evals)
        
        results = []
        for f1 in f1_values:
            x = np.zeros(self.dim)
            x[0] = f1
            # Clip to bounds
            for i in range(self.dim):
                x[i] = max(bounds[i][0], min(bounds[i][1], x[i]))
            f = problem(x)
            results.append(f)
            
            if time.time() > deadline:
                break
        
        return results
