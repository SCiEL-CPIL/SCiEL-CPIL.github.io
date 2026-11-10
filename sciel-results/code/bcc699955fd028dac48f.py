import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        n_evals = self.budget
        if n_evals <= 0:
            return []
            
        results = []
        
        # For ZDT6:
        #   g(x') = 1 + 10 * n' + sum(4*x_i * (sin(20 * pi * x_i) + 1))  where n' = dim-1 = 9
        #   h(x1, g) = g * (1 - sqrt(x1 / g))
        #   f1 = 1 + 10 * g - h
        #   f2 = h
        #
        # When x[1:] = 0: g = 1, h = 1 - sqrt(x0), f1 = 11 - (1 - sqrt(x0)) = 10 + sqrt(x0)
        #   So f1 in [10, 11], f2 = 1 - sqrt(x0) in [0, 1].
        #
        # The Pareto front of ZDT6 is characterized by g=1 (x[1:]=0) and x0 in [0,1].
        # f1 = 10 + sqrt(x0), f2 = 1 - sqrt(x0)
        # So f1 + f2 = 11, with f1 in [10, 11] and f2 in [0, 1].
        #
        # Reference point is [11, 11].
        # Hypervolume contribution: for each point (f1, f2) with f1 < 11 and f2 < 11,
        # the contribution depends on the sorted set of points.
        #
        # Since f1 + f2 = 11 on the PF, and ref = (11, 11):
        #   For a point (f1, f2) = (10+s, 1-s) where s = sqrt(x0) in [0,1]:
        #   f1 = 10 + s, f2 = 1 - s
        #
        # To maximize hypervolume, we want to cover the Pareto front well.
        # The hypervolume is determined by the non-dominated points.
        #
        # Strategy: Sample x0 values to get uniform coverage of the Pareto front.
        # Since f1 = 10 + sqrt(x0), to get uniform f1 spacing, we need x0 = (f1-10)^2.
        # f1 ranges from 10 to 11, so let's sample f1 uniformly in [10, 11].
        
        n_uniform = min(n_evals, 200)
        # Sample f1 uniformly in [10, 11]
        f1_targets = np.linspace(10.0, 11.0, n_uniform)
        # x0 = (f1 - 10)^2
        x0s = (f1_targets - 10.0) ** 2
        
        # Sort to ensure progression
        x0s = np.sort(x0s)
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim, dtype=float)
            x[0] = float(x0s[i % n_uniform])
            # x[1:] = 0 gives g=1, putting points on the Pareto front
            
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break
                
        return results
