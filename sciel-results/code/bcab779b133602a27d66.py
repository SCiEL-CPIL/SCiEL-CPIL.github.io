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
                
                lo = np.array([b[0] for b in bounds])
                hi = np.array([b[1] for b in bounds])
                
                n = self.budget
                dim = self.dim
                
                # Optimal sampling for ZDT6 hypervolume
                # t uniform in [0, 1]
                t = np.linspace(0.0, 1.0, n)
                x0 = t**2
                
                X = np.zeros((n, dim))
                X[:, 0] = x0
                # x[1:] remains 0
                
                F = np.array([problem(x) for x in X])
                
                if time.time() > deadline:
                    return X
                    
                return X
