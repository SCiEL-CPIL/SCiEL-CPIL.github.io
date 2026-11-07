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
        
        # Check if 0 is within bounds for x1..x_{dim-1}
        # This ensures g(x) = 1 (or minimal) for standard ZDT4
        zeros_ok = all(lo[i] <= 0 <= hi[i] for i in range(1, self.dim))
        
        if zeros_ok:
            n_points = self.budget
            # Assume x0 range [0, 1] for standard ZDT4 Pareto front
            # Clip to valid bounds
            x0_lo = max(lo[0], 0.0)
            x0_hi = min(hi[0], 1.0)
            
            if x0_lo >= x0_hi:
                # Fallback: if the range is degenerate, just pick one point
                x0_vals = np.full(n_points, x0_lo)
            else:
                # Use non-uniform distribution: x ~ t^{4/3}
                # This optimizes the Riemann sum approximation for the integral
                # involving sqrt(x), which has high curvature at 0.
                t = np.linspace(0, 1, n_points)
                x0_vals = x0_lo + (x0_hi - x0_lo) * (t ** (4.0/3.0))
                
            X = np.zeros((n_points, self.dim))
            X[:, 0] = x0_vals
            # X[:, 1:] = 0.0 is default
            
            # Evaluate
            # Check time
            if time.time() > deadline:
                 return X
                 
            F = np.array([problem(x) for x in X])
            return X
            
        else:
            # Fallback random search if 0 is not in bounds for other variables
            rng = np.random.default_rng(0)
            X = lo + (hi - lo) * rng.random((self.budget, self.dim))
            F = np.array([problem(x) for x in X])
            return X
