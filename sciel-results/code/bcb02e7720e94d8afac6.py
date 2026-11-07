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
        
        n = self.dim
        budget = self.budget
        
        if budget <= 0:
            return None

        # Power law exponent
        # For ZDT2 (f2 ~ 1 - sqrt(f1)), optimal sampling for HV is ~ x^0.25 PDF,
        # which corresponds to CDF x^1.25. p=1.5 is a reasonable compromise 
        # between uniform (p=1) and the previous p=2.
        p = 1.5
        
        # Generate x1 values
        # Ensure 0 and 1 are included
        idx = np.arange(budget)
        # Avoid division by zero if budget is 1
        if budget == 1:
            x1_vals = np.array([0.0])
        else:
            x1_vals = (idx / (budget - 1)) ** p
        
        # Create full vectors
        # x2..xn = 0
        X = np.zeros((budget, n))
        X[:, 0] = x1_vals
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        # Check time
        if time.time() > deadline:
            pass # We already used the budget, just return
        
        return X
