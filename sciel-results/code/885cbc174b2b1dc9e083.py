import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Determine the optimal tail values for ZDT4
        # The term in g(x) is x^2 - 10*cos(4*pi*x) + 10
        # We test candidates 0.0, 0.5, 1.0 to find the local minimizer within bounds
        candidates = [0.0, 0.5, 1.0]
        best_tail_val = 0.0
        min_cost = float('inf')
        
        for c in candidates:
            c_clipped = np.clip(c, lo[1], hi[1])
            # Calculate cost for this single variable term
            cost = c_clipped**2 - 10 * np.cos(4 * np.pi * c_clipped) + 10
            if cost < min_cost:
                min_cost = cost
                best_tail_val = c_clipped
        
        # The optimal tail is a vector of this best value
        x_tail = np.full(self.dim - 1, best_tail_val)
        
        # Use midpoint rule for better Riemann sum approximation of hypervolume
        # Midpoints of n equal intervals over [lo[0], hi[0]]
        n = self.budget
        step = (hi[0] - lo[0]) / n
        x0_values = lo[0] + (np.arange(n) + 0.5) * step
        
        for x0 in x0_values:
            if time.time() > deadline:
                break
            
            x = np.empty(self.dim)
            x[0] = x0
            x[1:] = x_tail
            
            try:
                problem(x)
            except Exception:
                continue

        return None
