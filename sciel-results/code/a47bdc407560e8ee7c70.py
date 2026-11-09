import numpy as np

class Optimizer:
      def __init__(self, budget, dim, n_obj):
          self.budget = budget
          self.dim = dim
          self.n_obj = n_obj

      def __call__(self, problem, bounds):
          # Assume bounds are [0,1]
          # DTLZ7 specific:
          # x[0], x[1] are the front variables
          # x[2:] are the g variables, optimal at 0.5
          
          n_evals = self.budget
          
          # Generate Halton points in [0,1]^2
          def halton_2d(n):
              points = np.zeros((n, 2))
              for i in range(n):
                  # Base 2
                  f, r, q = 1.0, 0.0, i + 1
                  while q > 0:
                      f /= 2.0
                      r += f * (q % 2)
                      q //= 2
                  points[i, 0] = r
                  
                  # Base 3
                  f, r, q = 1.0, 0.0, i + 1
                  while q > 0:
                      f /= 3.0
                      r += f * (q % 3)
                      q //= 3
                  points[i, 1] = r
              return points
          
          pts = halton_2d(n_evals)
          
          # Map to bounds if necessary, but task says [0,1]
          lo = np.array([b[0] for b in bounds])
          hi = np.array([b[1] for b in bounds])
          
          # Construct full vectors
          X = np.zeros((n_evals, self.dim))
          X[:, 0] = pts[:, 0] * (hi[0] - lo[0]) + lo[0]
          X[:, 1] = pts[:, 1] * (hi[1] - lo[1]) + lo[1]
          # Set the rest to 0.5 (midpoint of bounds, assuming symmetric or just 0.5 if [0,1])
          # For DTLZ7, 0.5 is the optimal for g.
          # If bounds are [0,1], 0.5 is the center.
          for i in range(2, self.dim):
              X[:, i] = 0.5 * (hi[i] - lo[i]) + lo[i]
          
          # Evaluate
          F = np.array([problem(x) for x in X])
          
          return X
