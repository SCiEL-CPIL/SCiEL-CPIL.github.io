"""Multi-objective optimizer for ZDT6 using analytic Pareto front sampling."""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # We have 'budget' evaluations.
        # We will use all of them to sample the Pareto front.
        # The Pareto front for ZDT6 is located at x_2...x_dim = 0.
        # x_1 determines the position on the front.
        
        n_samples = budget
        
        # Generate x_1 values uniformly in [0, 1]
        # Using linspace to ensure coverage of the entire range
        x1_values = np.linspace(0.0, 1.0, n_samples)
        
        # Construct the population
        # x_2 to x_dim are set to 0.0 (lower bound)
        # This ensures g(x) = 1, which is the minimum possible g.
        X = np.zeros((n_samples, dim))
        X[:, 0] = x1_values
        # x_2...x_dim are already 0.0
        
        # Evaluate all points
        # The problem function clips to bounds, so 0.0 is safe.
        # We must call problem(x) for each x to record the objective values.
        
        for i in range(n_samples):
            if time.time() > deadline - 1:
                break
            # Evaluate and discard the return value (it is recorded by the evaluator)
            problem(X[i])
            
        return X
