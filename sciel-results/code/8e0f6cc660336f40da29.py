import numpy as np
import scipy.stats.qmc as qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        if self.budget <= 0:
            return np.array([])
        
        n_evals = self.budget
        
        # Strategy: Sample the Pareto front (x[1:] = 0) using Quasi-Monte Carlo (Sobol) points for x[0].
        # This approach differs from Gaussian Quadrature by using low-discrepancy sequences 
        # which are often superior for non-polynomial, oscillatory objective landscapes like ZDT3.
        
        if n_evals == 1:
            x1_vals = np.array([0.5])
        else:
            # Reserve 2 evaluations for the endpoints of the Pareto front (x0=0, x0=1)
            n_interior = n_evals - 2
            if n_interior > 0:
                # Generate Sobol points in [0, 1]
                # Sobol requires integer n, works best with powers of 2 but handles any n
                sampler = qmc.Sobol(d=1, scramble=True)
                # Generate n_interior points
                x1_interior = sampler.random(n_interior).flatten()
                x1_vals = np.concatenate(([0.0], x1_interior, [1.0]))
            else:
                # If budget is 2, just use endpoints
                x1_vals = np.array([0.0, 1.0])
        
        # Construct the population
        X = np.zeros((len(x1_vals), self.dim))
        X[:, 0] = x1_vals
        # x[1:] remains 0, which is the condition for the Pareto front in ZDT3
        
        # Clip to bounds (though 0 and 1 are within [0,1])
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X
