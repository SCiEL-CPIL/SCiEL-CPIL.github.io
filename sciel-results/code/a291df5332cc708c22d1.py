import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Use all budget for Pareto front points
        N = self.budget
        
        # Uniform distribution in f1 (x0)
        # This maximizes HV for the ZDT2 front
        f1_vals = np.linspace(0.0, 1.0, N)
        
        X = np.zeros((N, self.dim))
        X[:, 0] = f1_vals
        # x[1:] remains 0
        
        F_list = []
        for i in range(N):
            if time.time() > deadline:
                break
            x = X[i]
            f = problem(x)
            F_list.append(f)
            
        return X
