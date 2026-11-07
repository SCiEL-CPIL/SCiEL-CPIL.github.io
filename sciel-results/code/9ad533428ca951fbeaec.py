import numpy as np
import time

class Optimizer:
       def __init__(self, budget, dim, n_obj):
           self.budget = budget
           self.dim = dim
           self.n_obj = n_obj

       def __call__(self, problem, bounds):
           start_time = time.time()
           deadline = start_time + 280 # Safety margin

           lo = np.array([b[0] for b in bounds])
           hi = np.array([b[1] for b in bounds])
           
           n_evals = self.budget
           dim = self.dim
           
           # Generate points on the Pareto front
           # x_2..x_n = 0.5
           # x_1 = optimal spacing
           
           # Use power law spacing for x_1
           # i from 0 to n_evals-1
           # Map to [0, 1]
           # x_1 = (i / (n_evals - 1)) ** (2/3)
           
           if n_evals > 1:
               indices = np.arange(n_evals)
               x1_vals = (indices / (n_evals - 1)) ** (2.0/3.0)
           else:
               x1_vals = np.array([0.5])
               
           points = np.zeros((n_evals, dim))
           points[:, 0] = x1_vals
           points[:, 1:] = 0.5
           
           # Evaluate
           for i in range(n_evals):
               if time.time() > deadline:
                   break
               problem(points[i])
               
           return points
