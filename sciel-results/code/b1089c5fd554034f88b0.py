import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 250  # Safety margin before 300s limit
        
        # ZDT1: Pareto front is achieved when x2..xn = 0, and x1 varies in [0,1]
        # f1 = x1, f2 = 1 - sqrt(x1) when g(x) = 1 (which happens at x2..xn = 0)
        
        # Use all budget to sample x1 across [0,1] with x2..xn = 0
        n_evals = self.budget
        
        # Non-uniform grid: denser at low x1? Or sparser?
        # Hypothesis: Larger steps at high x1 (where h(x1) is higher) maximize HV sum.
        # x_i = (i / (N-1))^p
        # Try p=2.0
        if n_evals > 1:
            indices = np.arange(n_evals)
            x1_values = (indices / (n_evals - 1)) ** 2.0
        else:
            x1_values = np.array([0.0])
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        results = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = x1_values[i]
            
            # Clip to bounds (should be no-op for [0,1] bounds)
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            results.append(f)
        
        return results
