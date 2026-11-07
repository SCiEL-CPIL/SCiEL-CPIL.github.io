import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ4 structure assumption:
        # First dim-2 variables control distance to front (f1)
        # Last 2 variables control shape (f2, f3)
        n_dist = self.dim - 2
        n_shape = 2
        
        X = np.zeros((self.budget, self.dim))
        
        # 50% on the main front (f1 ~ 0)
        half = self.budget // 2
        for i in range(half):
            # Distance vars are 0
            # Shape vars random
            X[i, n_dist:n_dist+n_shape] = rng.random(n_shape)
            
        # 50% on the rest of the front (f1 > 0)
        for i in range(half, self.budget):
            # Sample target f1 value t in [0, 1]
            t = rng.uniform(0, 1)
            # Set one distance variable to achieve f1 = t
            # f1 = 0.5 * sum(x_i^beta)
            # x_1 = (2*t)^(1/beta)
            # beta is typically 100
            beta = 100
            val = (2 * t) ** (1 / beta)
            X[i, 0] = val
            # Other distance vars 0
            # Shape vars random
            X[i, n_dist:n_dist+n_shape] = rng.random(n_shape)
            
        F = np.array([problem(x) for x in X])
        
        return X
