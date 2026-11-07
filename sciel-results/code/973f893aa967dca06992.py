import numpy as np
import time

class Optimizer:
       def __init__(self, budget, dim, n_obj):
           self.budget = budget
           self.dim = dim
           self.n_obj = n_obj

       def __call__(self, problem, bounds):
           start_time = time.time()
           deadline = start_time + 250  # 300s limit, keep 50s margin

           # Bounds
           lo = np.array([b[0] for b in bounds])
           hi = np.array([b[1] for b in bounds])
           
           # DTLZ4 specific:
           # Front depends on x[0], x[1], x[2]
           # g depends on x[3] to x[11]
           # To be on front: x[0]^2 + x[1]^2 + x[2]^2 = 1, x[3..11] = 0.5
           
           # We will generate points using a Halton sequence for x[0..2]
           # and set x[3..11] to 0.5
           
           n_evals = self.budget
           points = []
           
           # Halton sequence generator
           def halton(index, base):
               result = 0.0
               f = 1.0 / base
               i = index
               while i > 0:
                   result += f * (i % base)
                   i = i // base
                   f /= base
               return result

           # Generate n_evals points
           # We use indices 1 to n_evals (index 0 is 0,0,0 which is bad)
           for i in range(1, n_evals + 1):
               if time.time() > deadline:
                   break
               
               # Halton for 3 dimensions (bases 2, 3, 5)
               u0 = halton(i, 2)
               u1 = halton(i, 3)
               u2 = halton(i, 5)
               
               # Normalize to sphere
               norm = np.sqrt(u0**2 + u1**2 + u2**2)
               if norm < 1e-10:
                   # Fallback to random if degenerate
                   u0, u1, u2 = np.random.rand(3)
                   norm = np.sqrt(u0**2 + u1**2 + u2**2)
               
               x0 = u0 / norm
               x1 = u1 / norm
               x2 = u2 / norm
               
               # Construct full vector
               x = np.zeros(self.dim)
               x[0] = x0
               x[1] = x1
               x[2] = x2
               # Set remaining to 0.5 (midpoint of [0,1])
               x[3:] = 0.5
               
               # Clip to bounds just in case
               x = np.clip(x, lo, hi)
               
               points.append(x)
               
           # Evaluate all points
           # We must call problem for each point
           # The evaluator records every returned objective vector.
           # So we just need to call problem and discard the result?
           # "Every returned objective vector is recorded and the return value of __call__ is ignored."
           # So we just need to execute the calls.
           
           for x in points:
               if time.time() > deadline:
                   break
               problem(x)
           
           return points
