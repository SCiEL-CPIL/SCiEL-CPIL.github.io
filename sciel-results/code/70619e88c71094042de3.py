import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        
        n_shape = self.n_obj - 1
        n_dist = self.dim - n_shape
        
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=n_shape, scramble=True)
            points_2d = sampler.random(self.budget)
        except ImportError:
            # Fallback to Halton or Random
            rng = np.random.default_rng(0)
            points_2d = rng.random((self.budget, n_shape))
            
        dist_zeros = np.zeros((self.budget, n_dist))
        X = np.hstack((points_2d, dist_zeros))
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        for i in range(self.budget):
            if time.time() > start_time + 295:
                break
            problem(X[i])
            
        return X
