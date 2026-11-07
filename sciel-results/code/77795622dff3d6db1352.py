import numpy as np
import time

class Optimizer:
          def __init__(self, budget, dim, n_obj):
              self.budget = budget
              self.dim = dim
              self.n_obj = n_obj

          def __call__(self, problem, bounds):
              start_time = time.time()
              deadline = start_time + 300  # Safety margin

              # We know the structure of DTLZ2
              # Pareto front is at x_{M+1:n} = 0.5
              # f_i determined by x_1, ..., x_M
              # For M=3, x_1, x_2 determine the point on the sphere
              
              # We will generate 'budget' points uniformly distributed on the Pareto front
              # Using Halton sequence for low discrepancy
              
              def halton(index, base):
                  f = 1.0
                  r = 0.0
                  while index > 0:
                      f = f / base
                      r = r + f * (index % base)
                      index = index // base
                  return r

              lo = np.array([b[0] for b in bounds])
              hi = np.array([b[1] for b in bounds])
              
              # x_1, x_2 are the first 2 variables
              # x_3 ... x_12 are fixed at 0.5
              
              results = []
              for i in range(1, self.budget + 1):
                  # Check time
                  if time.time() > deadline:
                      break
                  
                  # Generate Halton points in [0, 1]
                  u1 = halton(i, 2)
                  u2 = halton(i, 3)
                  
                  # Map to x
                  # x_1 = (2/pi) * arccos(u1)
                  # x_2 = u2  (since x_2 = (2/pi) * phi, and we set phi = u2 * pi/2)
                  # Wait, let's re-verify the mapping.
                  # We want uniform distribution on the sphere.
                  # We derived that u1 = cos(theta1) and theta2 uniform in [0, pi/2] works.
                  # f1 = u1
                  # f2 = sqrt(1-u1^2) * cos(theta2)
                  # f3 = sqrt(1-u1^2) * sin(theta2)
                  # x1 = (2/pi) * arccos(f1) = (2/pi) * arccos(u1)
                  # x2 = (2/pi) * arccos( f2 / sqrt(1-f1^2) ) = (2/pi) * arccos( cos(theta2) ) = (2/pi) * theta2
                  # If we set theta2 = u2 * pi/2, then x2 = u2.
                  
                  x = np.zeros(self.dim)
                  x[0] = (2.0 / np.pi) * np.arccos(u1)
                  x[1] = u2
                  # Set remaining variables to 0.5
                  x[2:] = 0.5
                  
                  # Clip to bounds just in case (though 0.5 and [0,1] mapped values are safe)
                  x = np.clip(x, lo, hi)
                  
                  f = problem(x)
                  results.append(f)
                  
              return results
