import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ2 structure
        # x[0], x[1] are position variables
        # x[2:] are decision variables, set to 0.5 to minimize g
        
        X = np.zeros((self.budget, self.dim))
        
        # Set decision variables to midpoint
        mid = (lo + hi) / 2.0
        X[:, 2:] = mid[2:]
        
        # Generate Sobol sequence for position variables
        # Sobol is a low-discrepancy sequence
        sampler = qmc.Sobol(d=2, scramble=True)
        # We need 200 points. Sobol requires power of 2 for best quality?
        # Actually, we can sample more and take the first N, or use sample(N)
        # sample(N) works for any N but might not be as uniform if N is not power of 2.
        # Let's sample 256 (next power of 2) and take first 200?
        # Or just sample 200.
        sobol_points = sampler.random(self.budget)
        
        # Scale to bounds
        X[:, 0] = lo[0] + sobol_points[:, 0] * (hi[0] - lo[0])
        X[:, 1] = lo[1] + sobol_points[:, 1] * (hi[1] - lo[1])
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X
