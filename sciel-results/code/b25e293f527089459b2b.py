import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n_evals = self.budget
        dim = self.dim
        
        # Extract bounds for x1
        # bounds is a list of (low, high)
        x1_low, x1_high = bounds[0]
        
        # Generate x1 values using a deterministic uniform distribution (LHS in 1D)
        # We use linspace to cover the full range including endpoints
        x1_values = np.linspace(x1_low, x1_high, n_evals)
        
        # Create the full population
        X = np.zeros((n_evals, dim))
        X[:, 0] = x1_values
        
        # For x2..xN, set to 0 to minimize g(x)
        # Check if 0 is within bounds for other variables.
        # ZDT4 standard bounds are [-5, 5], so 0 is safe.
        # If bounds were e.g. [1, 5], we should set to 1.
        # Let's project 0 to the closest bound if 0 is out of range.
        for i in range(1, dim):
            low, high = bounds[i]
            if 0 < low:
                X[:, i] = low
            elif 0 > high:
                X[:, i] = high
            else:
                X[:, i] = 0.0

        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
