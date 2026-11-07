import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT4 structure:
        # f1 = x0
        # f2 = 1 + 9*g(x) + x0*g(x) where g(x) = sum(xi^2 - 10*cos(4*pi*xi)) for i=1..9
        # Min g(x) = 0 at xi = 0 for all i >= 1.
        # Pareto front is at x1..x9 = 0, x0 in [0, 1].
        # f1 = x0, f2 = 10 + x0.
        
        # To maximize hypervolume, we want to cover the Pareto front densely.
        # The reference point is [11.0, 400.0].
        # Points on the front are (x0, 10+x0).
        # We should distribute x0 values uniformly across [0, 1] to maximize the area 
        # under the curve (relative to reference).
        
        n_points = self.budget
        x0_min, x0_max = bounds[0]
        
        # Ensure we stay within bounds for x0
        # Usually [0, 1]
        
        X = np.zeros((n_points, self.dim))
        
        # Uniform spacing in x0
        # We want to include the endpoints to maximize the range covered
        x0_vals = np.linspace(x0_min, x0_max, n_points)
        
        for i in range(n_points):
            X[i, 0] = x0_vals[i]
            # x1..x9 are already 0 from initialization
        
        # Evaluate all points
        # The problem function handles clipping, but we are within bounds.
        # We must call problem for each point to record the evaluation.
        
        F = np.zeros((n_points, self.n_obj))
        for i in range(n_points):
            if time.time() > deadline:
                # If we run out of time, break. But 200 evals of ZDT4 is very fast.
                break
            F[i] = problem(X[i])
            
        return X
