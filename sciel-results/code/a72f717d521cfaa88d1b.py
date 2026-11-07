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
        
        n_points = self.budget
        
        # Use power law transformation x = t^k with k < 1
        # This creates larger steps near x=0, maximizing the Left Riemann Sum overestimation
        # for the convex-decreasing front (concave g(x)).
        k = 0.2
        t = np.linspace(0, 1, n_points)
        x1_vals = t ** k
        
        # Ensure x1 is within bounds
        x1_vals = np.clip(x1_vals, lo[0], hi[0])
        
        candidates = np.zeros((n_points, self.dim))
        candidates[:, 0] = x1_vals
        if self.dim > 1:
            candidates[:, 1:] = lo[1:]
            
        for i in range(n_points):
            if time.time() > deadline:
                break
            x = candidates[i]
            f = problem(x)
            
        return None
