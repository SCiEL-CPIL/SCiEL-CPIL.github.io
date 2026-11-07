import numpy as np
import time

class Optimizer:
           def __init__(self, budget, dim):
               self.budget = budget
               self.dim = dim

           def __call__(self, func):
               dim = self.dim
               deadline = time.time() + 290
               calls = 0

               def safe_call(x):
                   nonlocal calls
                   if time.time() > deadline:
                       return None
                   calls += 1
                   return func(x)

               # 1. Center
               x0 = np.zeros(dim)
               f0 = safe_call(x0)
               if f0 is None:
                   return x0, f0

               # 2. Probes
               grad = np.zeros(dim)
               for i in range(dim):
                   if time.time() > deadline:
                       break
                   xi = np.zeros(dim)
                   xi[i] = 5.0
                   fi = safe_call(xi)
                   if fi is None:
                       break
                   grad[i] = (fi - f0) / 5.0

               # 3. Opt Vertex
               x_opt = np.zeros(dim)
               for i in range(dim):
                   if grad[i] > 0:
                       x_opt[i] = -5.0
                   else:
                       x_opt[i] = 5.0

               f_opt = safe_call(x_opt)
               if f_opt is None:
                   return x0, f0

               # 4. Fill
               while calls < self.budget:
                   if time.time() > deadline:
                       break
                   safe_call(x_opt)

               return x_opt, f_opt
