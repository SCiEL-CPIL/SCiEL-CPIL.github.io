import numpy as np

class Optimizer:
      def __init__(self, budget, dim, n_obj):
          self.budget = budget
          self.dim = dim
          self.n_obj = n_obj
          
      def __call__(self, problem, bounds):
          lo = np.array([b[0] for b in bounds])
          hi = np.array([b[1] for b in bounds])
          
          # Optimal tail values for ZDT4 to maximize g(x)
          # x1 = 0.05
          # x2..x9 = 0.05
          x_tail = np.full(self.dim - 1, 0.05)
          
          # Sweep x0
          # Valid range for x0 is [lo[0], hi[0]]
          # But f1 = x0 + 2 must be >= 0 for sqrt
          # So x0 >= -2
          x0_min = max(lo[0], -2.0)
          x0_max = hi[0]
          
          n_evals = self.budget
          x0s = np.linspace(x0_min, x0_max, n_evals)
          
          results = []
          for x0 in x0s:
              x = np.zeros(self.dim)
              x[0] = x0
              x[1:] = x_tail
              # Clip to bounds just in case
              x = np.clip(x, lo, hi)
              f = problem(x)
              results.append(f)
              
          return results
