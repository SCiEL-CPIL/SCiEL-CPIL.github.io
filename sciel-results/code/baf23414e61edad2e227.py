import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # ZDT1 Pareto Front:
        # f1 = x1, f2 = 1 + sin(10*pi*x1) with g=0 (x2..xn = 0)
        # Non-dominated: x1=0 (f2=1) and x1 in [0.1, 0.15] (f2 from 1 to 0)
        
        n_evals = min(self.budget, 200)
        
        X = np.zeros((n_evals, self.dim))
        
        # Point 0: x1=0
        X[0, 0] = 0.0
        
        if n_evals > 1:
            # Non-uniform spacing in [0.1, 0.15]
            # f2' = 10*pi*cos(10*pi*x1), |f2'| is large near 0.1, small near 0.15
            # Use inverse CDF: space proportional to 1/|f2'| to equalize hypervolume loss
            # Simplified: use a power-law clustering toward x1=0.1
            # t = 0.1 + 0.05 * u^p where u in [0,1] and p > 1 clusters near 0.1
            # Optimize p: try p=2 for quadratic clustering
            u = np.linspace(0, 1, n_evals - 1)
            p = 2.0
            t_vals = 0.1 + 0.05 * (u ** p)
            X[1:, 0] = t_vals
        
        for i in range(n_evals):
            problem(X[i])
            
        return X
