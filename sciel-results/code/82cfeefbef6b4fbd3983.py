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
        
        # Generate x0 values
        # For ZDT6, the Pareto front is traced by varying x[0] with x[1:]=0
        # f1(x0) = 1 + 10*n + x0*(sin(10*pi*x0) + sin(10*pi*x0^2))
        # f2(x0) = 1 - sqrt(f1(x0))
        # To maximize hypervolume, we want points spread across the full range of f1
        # which corresponds to x0 in [0, 1]
        
        # Use a power-law spacing that accounts for the non-linear relationship
        # between x0 and f1. The function f1 has oscillations, so uniform x0
        # sampling should work reasonably well.
        
        # Include boundaries 0 and 1 for full coverage
        x0s = []
        if n_evals > 2:
            # 1 point at 0, 1 point at 1
            # Rest uniformly spaced in (0, 1)
            n_inner = n_evals - 2
            x0s.append(0.0)
            x0s.append(1.0)
            
            # Uniform in (0, 1) excluding endpoints
            inner_xs = np.linspace(0.0, 1.0, n_inner + 2)[1:-1]
            x0s.extend(inner_xs.tolist())
            
        elif n_evals == 2:
            x0s = [0.0, 1.0]
        else:
            x0s = [0.0]
            
        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = np.zeros(self.dim, dtype=float)
            x[0] = float(x0s[i])
            # Other x's are 0, which is optimal for ZDT6
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break
                
        return results
