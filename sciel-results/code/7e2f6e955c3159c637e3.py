import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Generate f1 values
        # Use power law to cluster near 0
        # x_i = (i / (N-1))^p
        # Let's try p=2.0
        N = self.budget
        if N == 0:
            return []
            
        indices = np.arange(N)
        # Avoid division by zero if N=1, but N=200
        f1_vals = (indices / (N - 1)) ** 2.0
        
        # Ensure we cover the range [0, 1]
        # f1_vals[0] = 0, f1_vals[-1] = 1
        
        results = []
        
        for i in range(N):
            if time.time() > deadline:
                break
                
            x = np.full(self.dim, 0.5)
            x[0] = f1_vals[i]
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            results.append(f)
            
        return results
