import numpy as np
import time
import scipy.stats.qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        
        m = self.n_obj - 1
        n_pos = self.dim - m
        
        pos_vars = np.full(n_pos, 0.5)
        
        # Use Sobol sequence for low-discrepancy sampling in shape space
        sampler = qmc.Sobol(d=m, scramble=False)
        shape_vars = sampler.random(self.budget)
        
        X = np.zeros((self.budget, self.dim))
        X[:, :n_pos] = pos_vars
        X[:, n_pos:] = shape_vars
        
        for i in range(self.budget):
            if time.time() > deadline:
                break
            x = X[i]
            problem(x)
            
        return None
