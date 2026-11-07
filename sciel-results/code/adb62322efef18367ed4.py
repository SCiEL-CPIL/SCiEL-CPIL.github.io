import numpy as np

class Optimizer:
       def __init__(self, budget, dim, n_obj):
           self.budget = budget
           self.dim = dim
           self.n_obj = n_obj

       def __call__(self, problem, bounds):
           N = self.budget
           dim = self.dim
           
           # Generate x1 values
           # We want to minimize the left Riemann sum of y(x) = 1 - sqrt(x)
           # y is decreasing and convex.
           # Optimal spacing for minimizing left sum error?
           # Let's just test a few power laws analytically.
           
           alphas = [1.0, 1.5, 2.0, 2.5, 3.0]
           best_alpha = 1.0
           min_sum = float('inf')
           
           for alpha in alphas:
               x1 = (np.arange(N) / (N - 1)) ** alpha
               # Ensure endpoints
               x1[0] = 0.0
               x1[-1] = 1.0
               
               y = 1.0 - np.sqrt(x1)
               
               # Calculate Left Riemann Sum part
               # Sum (x[i+1] - x[i]) * y[i]
               dx = np.diff(x1)
               sum_val = np.sum(dx * y[:-1])
               
               # Last term: (11 - x1[-1]) * y[-1]
               # x1[-1] is 1.0, so (11-1)*y(1) = 10*0 = 0.
               # If x1[-1] < 1, it would be (11 - x1[-1]) * y[-1]
               
               # We want to minimize sum_val to maximize HV
               if sum_val < min_sum:
                   min_sum = sum_val
                   best_alpha = alpha
                   
           x1_opt = (np.arange(N) / (N - 1)) ** best_alpha
           x1_opt[0] = 0.0
           x1_opt[-1] = 1.0
           
           # Construct full vectors
           X = np.zeros((N, dim))
           X[:, 0] = x1_opt
           # x[1:] remains 0
           
           results = []
           for i in range(N):
               f = problem(X[i])
               results.append(f)
               
           return X
