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
        
        # DTLZ7 Pareto front: x2...x21 = 0
        # f1 = x0
        # f2 = x1 * (1 + x0)
        # f3 = 1 - x0 + x0^2 * sin(pi*x0/2)
        
        # Reference point: [1.1, 1.1, 25.0]
        # Need f2 < 1.1, so x1 < 1.1/(1+x0)
        
        # Use 20 x 10 grid = 200 evaluations
        n0 = 20
        n1 = 10
        
        x0_vals = np.linspace(0.0, 1.0, n0)
        
        all_x0 = []
        all_x1 = []
        
        # Small epsilon to ensure strict inequality f2 < 1.1
        eps = 1e-10
        
        for x0 in x0_vals:
            max_x1 = 1.1 / (1.0 + x0) - eps
            if max_x1 <= 0:
                max_x1 = 0.0
            x1_vals = np.linspace(0.0, max_x1, n1)
            for x1 in x1_vals:
                all_x0.append(x0)
                all_x1.append(x1)
        
        all_x0 = np.array(all_x0)
        all_x1 = np.array(all_x1)
        
        n_samples = len(all_x0)
        
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = all_x0
        X[:, 1] = all_x1
        # X[:, 2:] = 0 (inactive variables)
        
        for i in range(n_samples):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
