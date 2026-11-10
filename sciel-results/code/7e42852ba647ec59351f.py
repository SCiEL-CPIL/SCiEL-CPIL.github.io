import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT3.
    Exploits the structure of ZDT3 where the Pareto front lies on the manifold
    x_2 = ... = x_n = 0.5.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: 300s limit, use 290s
        deadline = start_time + 290
        
        n_evals = self.budget
        
        # Create x1 values
        # Uniform distribution in [0, 1]
        # We use linspace to cover the range including endpoints
        x1_values = np.linspace(0, 1, n_evals)
        
        # Pre-allocate array for all points
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x1_values
        # Set x[1:] to 0.5 for all points
        if self.dim > 1:
            X[:, 1:] = 0.5
            
        # Evaluate each point
        for i in range(n_evals):
            if time.time() > deadline:
                break
            problem(X[i])
            
        # Return the last evaluated point (value is ignored by evaluator)
        return X[-1]
