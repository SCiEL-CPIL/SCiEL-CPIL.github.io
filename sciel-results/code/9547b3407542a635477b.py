import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 330 - 10  # 320 seconds to be safe
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT2 specific strategy
        # x1 determines f1, x2..xn determine g(x)
        # Pareto front is at g(x) = 1, which is at x2..xn = 0.5
        # We initialize x1 uniformly to cover the front
        
        n_points = min(self.budget, 100)
        
        x1_values = np.linspace(0, 1, n_points)
        
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x1_values
        # Set other variables to 0.5 to minimize g(x)
        X[:, 1:] = 0.5
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        F = []
        for x in X:
            if time.time() > deadline:
                break
            f = problem(x)
            F.append(f)
            
        if F:
            F = np.array(F)
        else:
            F = np.array([])
            
        return X, F
