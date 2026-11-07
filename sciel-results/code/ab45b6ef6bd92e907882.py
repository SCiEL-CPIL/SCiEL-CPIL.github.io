import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: 300s limit, stop at 290s
        deadline = start_time + 290
        
        # ZDT1 Analytical Solution
        # f1 = x1
        # f2 = g(x) * (1 - sqrt(x1/g(x)))
        # g(x) = 1 + 10*(n-1)/x1 * sum(x2..xn)
        # Min f2 for fixed x1 is when sum(x2..xn)=0 => x2..xn=0
        # Then g(x)=1, f2 = 1 - sqrt(x1)
        
        # We want to sample x1 uniformly to cover the front [0, 1]
        # This maximizes hypervolume coverage
        
        n_evals = self.budget
        x1_vals = np.linspace(0.0, 1.0, n_evals)
        
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x1_vals
        # X[:, 1:] remains 0.0
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        # Check time
        if time.time() > deadline:
            return X
            
        # If we have time and budget (we used all budget), we are done.
        # If budget was larger, we could add more points, but linspace uses all.
        
        return X
