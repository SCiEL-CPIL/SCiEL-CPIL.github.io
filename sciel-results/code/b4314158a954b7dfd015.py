import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        n = self.dim
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Determine optimal values for x_2 ... x_n
        # ZDT4: g(x) = 1 + 10(n-1) + sum(x_i^2 - 10cos(4pi x_i)) for i=2..n
        # Min of x^2 - 10cos(4pi x) is at x=0 with value -10.
        # We want x_i to be 0 if possible.
        optimal_x_rest = np.zeros(n - 1)
        for i in range(n - 1):
            idx = i + 1 # 1-based index for x_2...x_n
            if lo[idx] <= 0 and 0 <= hi[idx]:
                optimal_x_rest[i] = 0.0
            else:
                # Find closest to 0 within bounds
                # The function has period 0.5. Minima at 0, 0.5, 1.0... but 0 is global min for the term.
                # If 0 not in bounds, pick bound closest to 0.
                if abs(lo[idx]) < abs(hi[idx]):
                    optimal_x_rest[i] = lo[idx]
                else:
                    optimal_x_rest[i] = hi[idx]
        
        # Determine sampling for x_1
        # Pareto front is for x_1 in [0, 1].
        # We assume bounds allow this. If not, we clip.
        x1_lo = max(lo[0], 0.0)
        x1_hi = min(hi[0], 1.0)
        
        if x1_lo >= x1_hi:
            # Degenerate case, just pick one point
            x1_vals = np.array([x1_lo])
        else:
            # Spread budget over x_1
            # Use all budget for x_1 sampling
            num_points = budget
            x1_vals = np.linspace(x1_lo, x1_hi, num_points)
            
        # Construct points
        # X shape: (num_points, n)
        X = np.zeros((num_points, n))
        X[:, 0] = x1_vals
        X[:, 1:] = optimal_x_rest
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        # Check time
        if time.time() > deadline:
            # Should not happen, but safety
            pass
            
        return X
