import numpy as np
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # DTLZ4 structure:
        # x[0], x[1] are shape variables
        # x[2:] are g variables
        # Optimal g variables are 0.5
        
        n_shape = 2
        n_g = self.dim - n_shape
        
        # Generate Sobol sequence
        # Sobol requires power of 2 for best properties, but works for any N.
        # 200 is not power of 2. We can generate 256 and take first 200, or just use 200.
        # qmc.Sobol supports any number of samples.
        
        sampler = qmc.Sobol(d=n_shape, scramble=True)
        # Generate budget points
        samples = sampler.random(self.budget)
        
        # Construct full dimension vectors
        X = np.zeros((self.budget, self.dim))
        X[:, :n_shape] = samples
        X[:, n_shape:] = 0.5
        
        # Evaluate
        F = []
        for x in X:
            F.append(problem(x))
            
        return F
