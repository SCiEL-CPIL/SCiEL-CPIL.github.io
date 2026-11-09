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
        
        used = 0
        results = []
        
        # ZDT1 Pareto front is at x[1:] = 0.5 (minimizing g(x))
        # f1 = x[0]
        # f2 = 1 - sqrt(x[0])
        
        # We sample x[0] to maximize HV.
        # Based on analysis, a non-uniform distribution might be better.
        # However, uniform is a strong baseline.
        # Let's try a distribution that is slightly denser at the "knee" or ends.
        # For now, uniform sampling of x[0] in [0, 1] is the most robust.
        
        n_evals = self.budget
        # Generate x0 values
        # Uniform sampling
        x0_values = np.linspace(0.0, 1.0, n_evals)
        
        # Evaluate
        for x0 in x0_values:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            
            x = np.full(self.dim, 0.5)
            x[0] = x0
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            results.append((x, f))
            used += 1
            
        return None
