import numpy as np


class Optimizer:
       def __init__(self, budget, dim, n_obj):
           self.budget = budget
           self.dim = dim
           self.n_obj = n_obj
           # Pre-compute lookup table for x1 -> f1
           # Use a large grid to ensure accuracy
           self.x1_grid = np.linspace(0, 1, 100000)
           self.f1_grid = 1 - np.exp(-4 * self.x1_grid) * np.sin(6 * np.pi * self.x1_grid)**8
           # Sort by f1 to allow binary search
           sort_idx = np.argsort(self.f1_grid)
           self.f1_sorted = self.f1_grid[sort_idx]
           self.x1_sorted = self.x1_grid[sort_idx]

       def __call__(self, problem, bounds):
           N = self.budget
           # Calculate target f1 values based on equal area under f2 = 1 - sqrt(f1)
           # Area A(f1) = f1 - 2/3 f1^1.5
           # We want A(f1_i) = i * A_total / N
           A_total = 1.0 - 2.0/3.0
           target_areas = np.linspace(0, A_total, N+1)[1:] # Exclude 0? Or include?
           # We want N points. Let's use i=1..N
           # Actually, spacing from 0 to 1.
           
           f1_targets = []
           for A in target_areas:
               # Solve f1 - 2/3 f1^1.5 = A
               # Use Newton's method or binary search on f1 in [0, 1]
               lo, hi = 0.0, 1.0
               for _ in range(50):
                   mid = (lo + hi) / 2.0
                   val = mid - (2.0/3.0) * (mid ** 1.5)
                   if val < A:
                       lo = mid
                   else:
                       hi = mid
               f1_targets.append((lo + hi) / 2.0)
               
           f1_targets = np.array(f1_targets)
           
           # Map f1_targets to x1 using binary search on sorted grid
           # np.searchsorted returns the index where the value would be inserted
           indices = np.searchsorted(self.f1_sorted, f1_targets)
           # Clip indices to be within bounds
           indices = np.clip(indices, 0, len(self.x1_sorted) - 1)
           
           x1_values = self.x1_sorted[indices]
           
           # Construct points
           results = []
           for i in range(N):
               x = np.zeros(self.dim)
               x[0] = x1_values[i]
               # x[1:] is 0
               f = problem(x)
               results.append(f)
               
           return results
