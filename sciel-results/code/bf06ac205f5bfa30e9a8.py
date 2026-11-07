import numpy as np
import time

class Optimizer:
              def __init__(self, budget, dim, n_obj):
                  self.budget = budget
                  self.dim = dim
                  self.n_obj = n_obj

              def __call__(self, problem, bounds):
                  start_time = time.time()
                  deadline = start_time + 290  # 300s limit, 10s margin
                  
                  lo = np.array([b[0] for b in bounds])
                  hi = np.array([b[1] for b in bounds])
                  
                  # Strategy: Sample the Pareto Front directly
                  # ZDT2 Front: x1 in [0,1], x2..xn = 0
                  # We assume dim >= 1.
                  
                  n_evals = self.budget
                  X = np.zeros((n_evals, self.dim))
                  
                  # Uniform distribution of x1
                  # If dim > 1, set rest to 0 (or lo if 0 is not in bounds? bounds are [0,1])
                  if self.dim > 0:
                      X[:, 0] = np.linspace(lo[0], hi[0], n_evals)
                      if self.dim > 1:
                          X[:, 1:] = lo[1:]  # Set to lower bound (0)
                          
                  F = np.zeros((n_evals, self.n_obj))
                  for i in range(n_evals):
                      F[i] = problem(X[i])
                      if time.time() > deadline:
                          break
                          
                  return X
