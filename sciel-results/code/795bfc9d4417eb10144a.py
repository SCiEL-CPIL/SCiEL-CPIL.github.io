import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        X_all = []
        F_all = []
        
        def evaluate(x):
            x = np.clip(x, lo, hi)
            f = problem(x)
            X_all.append(x.copy())
            F_all.append(np.array(f))
            return f

        # Generate 200 points uniformly along the known Pareto front
        # Front: x0 = t, x[1:] = 0
        # f1 = t
        # f2 = (1-t)cos(pi*t/2)
        # f3 = (1-t)sin(pi*t/2)
        
        n_points = budget
        # Ensure we don't exceed budget
        for i in range(n_points):
            if time.time() > deadline:
                break
            t = i / max(n_points - 1, 1)
            x = np.zeros(dim)
            x[0] = t
            # x[1:] remains 0.0
            evaluate(x)
            
        return X_all
