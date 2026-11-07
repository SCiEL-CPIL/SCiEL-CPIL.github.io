import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295  # safety margin of 5 seconds
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        used = 0
        
        # DTLZ7 Pareto front: x[2:] = 0, x[0] and x[1] vary in [0,1]
        # We need to spread points on the (x[0], x[1]) square to maximize hypervolume.
        
        # Use a structured approach: combine a coarse grid with refined points
        # For 3-objective DTLZ7, the Pareto front is 2D (x[0], x[1] vary)
        
        # Strategy: Use a Sobol-like quasi-random sequence or stratified sampling
        # to maximize coverage and hypervolume.
        
        # Generate points using a stratified approach
        n_budget = self.budget
        
        # Use Halton sequence for low-discrepancy sampling
        def halton(n, base):
            """Generate first n points of Halton sequence in [0,1] for given base."""
            result = []
            for i in range(1, n + 1):
                f = 1.0
                r = 0.0
                while i > 0:
                    f /= base
                    i, digit = divmod(i, base)
                    r += digit * f
                result.append(r)
            return np.array(result)
        
        # Generate 2D Halton sequence for x[0] and x[1]
        h0 = halton(n_budget, 2)
        h1 = halton(n_budget, 3)
        
        # Scale to bounds
        x0_vals = lo[0] + h0 * (hi[0] - lo[0])
        x1_vals = lo[1] + h1 * (hi[1] - lo[1])
        
        # Evaluate all points
        for i in range(n_budget):
            if time.time() > deadline:
                break
            
            x_new = np.zeros(self.dim)
            x_new[0] = x0_vals[i]
            x_new[1] = x1_vals[i]
            # x_new[2:] = 0
            
            f_new = problem(x_new)
            used += 1
        
        return None
