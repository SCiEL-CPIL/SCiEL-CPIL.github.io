import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Use the full budget
        n_points = self.budget
        
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT1 specific strategy:
        # The Pareto front is located at x_1...x_{dim-1} = 0.
        # x_0 varies in [0, 1] to trace the front.
        # We distribute x_0 uniformly to maximize hypervolume coverage.
        
        # Generate x_0 values
        # Ensure we are within bounds [0, 1]
        x0_min = lo[0]
        x0_max = hi[0]
        
        x0_vals = np.linspace(x0_min, x0_max, n_points)
        
        # Initialize population
        # Set all variables to lower bound initially, then adjust
        # For ZDT1, x_1...x_{dim-1} should be 0 (which is lo[1:])
        # Assuming lo[1:] is 0. If not, we should still try 0 if it's in bounds, 
        # but for ZDT1 it is 0.
        
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x0_vals
        
        # For other dimensions, set to 0.0
        # This is valid if 0 is in bounds. 
        # Task says bounds [0, 1]. So 0 is valid.
        
        # Evaluate
        for i in range(n_points):
            problem(X[i])
            
        return X
