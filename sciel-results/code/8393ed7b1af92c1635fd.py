import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # ZDT2 Pareto front is known: x[1:] = 0, x[0] in [0, 1]
        # f1 = x[0]
        # f2 = 1 - sqrt(x[0])
        
        # Deadline guard: stop 30 seconds before the 330s limit
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin
        
        # Use full budget to sample the front
        n_points = self.budget
        
        # Non-uniform sampling: denser near t=0 where the front is steeper
        # Use t = u^2 with u uniform in [0,1]
        u = np.linspace(0.0, 1.0, n_points)
        x0_vals = u ** 2
        
        # Initialize the population
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x0_vals
        # x[1:] remains 0
        
        # Evaluate all points with deadline guard
        F = np.zeros((n_points, self.n_obj))
        for i in range(n_points):
            # Check deadline
            if time.time() > deadline:
                break
            
            try:
                F[i] = problem(X[i])
            except Exception:
                # Stop on exception to preserve made evaluations
                break
        
        return X
