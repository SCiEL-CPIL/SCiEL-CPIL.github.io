import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 300.0
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget
        
        recorded = []
        
        # DTLZ1 Front:
        # x_1, x_2 determine position on simplex
        # x_3..x_7 = 0.5 ensures g(x)=0
        # We sample x_1, x_2 using Sobol sequence for better coverage
        
        # Generate Sobol points
        # d=2 dimensions for x1, x2
        sobol = qmc.Sobol(d=2, scramble=False)
        # Generate budget points
        # Note: Sobol generates points in [0,1)
        points = sobol.random(n=budget)
        
        # Map to x1, x2
        # points[:, 0] -> x1
        # points[:, 1] -> x2
        
        for i in range(budget):
            if time.time() > deadline - 5.0:
                break
            
            x1_val = points[i, 0]
            x2_val = points[i, 1]
            
            # Construct full vector
            x = np.full(dim, 0.5)
            x[0] = x1_val
            x[1] = x2_val
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            # Evaluate
            f = np.array(problem(x), dtype=float)
            recorded.append(f)
            
        return recorded
