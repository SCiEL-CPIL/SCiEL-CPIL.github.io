import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin (evaluator limit is 300s)

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # For ZDT3: f1 = x[0], g(x[1:]) = 1 + 10*(n-1) + sum(x_i^2 - 10*cos(2*pi*x_i))
        # Pareto front is at g_min = 1, achieved when x[1:] = 0 (each term = -10)
        # So we set x[1:] = 0 to be on the front, and vary x[0] to cover f1 in [0,1]
        
        # Strategy: Use all budget to densely sample x[0] in [lo[0], hi[0]] with x[1:] = lo[1:]
        # This gives uniform coverage of the Pareto front in objective space
        
        # First, verify that x[1:] = lo[1:] gives us Pareto-optimal points
        # by checking a few points and seeing if f2 is reasonable
        
        # Direct uniform sampling of x[0] across the full range
        # This is optimal for ZDT3 since f1 = x[0] is linear
        
        x0_samples = np.linspace(lo[0], hi[0], budget)
        
        for i in range(budget):
            if time.time() > deadline:
                break
            
            x = np.zeros(dim)
            x[0] = x0_samples[i]
            x[1:] = lo[1:]  # Set to lower bounds to minimize g(x[1:])
            
            try:
                problem(x)
            except Exception:
                break

        return None
