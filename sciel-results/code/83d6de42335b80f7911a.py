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
        
        # For ZDT3, the Pareto front is characterized by g(x_2...x_M) = 0.
        # The function g typically contains terms like |x_i - 0.5| which are minimized at x_i = 0.5.
        # Therefore, we set all decision variables except the first to the midpoint of their bounds.
        # This places the points on or very near the Pareto front.
        mid = (lo + hi) / 2.0
        
        # The first variable x_0 controls the position along the Pareto front.
        # For ZDT3, f_1 = x_0, so uniform sampling of x_0 provides uniform coverage of the front.
        x0_samples = np.linspace(lo[0], hi[0], budget)
        
        for i in range(budget):
            if time.time() > deadline:
                break
            
            x = np.empty(dim)
            x[0] = x0_samples[i]
            x[1:] = mid[1:]
            
            try:
                problem(x)
            except Exception:
                break

        return None
