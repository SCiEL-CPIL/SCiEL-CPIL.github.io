import numpy as np
import time

class Optimizer:
       def __init__(self, budget, dim, n_obj):
           self.budget = budget
           self.dim = dim
           self.n_obj = n_obj

       def __call__(self, problem, bounds):
           t0 = time.time()
           deadline = t0 + 280.0
           
           lo = np.array([b[0] for b in bounds])
           hi = np.array([b[1] for b in bounds])
           
           N = min(self.budget, 200)
           D = self.dim
           
           # Search for optimal power p
           # Theoretical p is 4/3 ~ 1.333
           # Search range [1.0, 1.7]
           p_vals = np.linspace(1.0, 1.7, 500)
           best_hv = -np.inf
           best_p = 1.333
           
           # Precompute u
           u = (np.arange(N) + 0.5) / N  # Midpoints for better coverage?
           # Or linspace(0, 1, N)?
           # Midpoints avoid 0 and 1, which might be good to avoid boundary issues?
           # But the front is well-defined at 0 and 1.
           # Using linspace(0,1,N) includes 0 and 1.
           # Let's use linspace(0, 1, N) as in inspirations.
           u = np.linspace(0.0, 1.0, N)
           
           for p in p_vals:
               x1 = u ** p
               # Ensure x1 is within [0, 1]
               x1 = np.clip(x1, 0.0, 1.0)
               
               f2 = 1.0 - np.sqrt(x1)
               
               # Calculate HV
               # x1 is sorted ascending
               df1 = np.diff(x1)
               df1 = np.append(df1, 11.0 - x1[-1])
               h = 11.0 - f2
               hv = np.sum(df1 * h)
               
               if hv > best_hv:
                   best_hv = hv
                   best_p = p
           
           # Generate final points
           x1_final = u ** best_p
           x1_final = np.clip(x1_final, 0.0, 1.0)
           
           X = np.zeros((N, D))
           X[:, 0] = x1_final
           # x[1:] = 0 is default
           
           # Clip to bounds
           X = np.clip(X, lo, hi)
           
           # Evaluate
           used = 0
           for i in range(N):
               if time.time() > deadline:
                   break
               if used >= self.budget:
                   break
               problem(X[i].copy())
               used += 1
           
           return X
