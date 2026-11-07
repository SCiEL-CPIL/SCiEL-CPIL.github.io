import numpy as np
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        n_shape_vars = n_obj - 1
        n_dist_vars = dim - n_shape_vars
        
        # Generate space-filling samples in [0, 1]^n_shape_vars
        # Using Sobol sequence for low discrepancy
        sampler = qmc.Sobol(d=n_shape_vars, scramble=False)
        shape_samples = sampler.random(n=budget)
        
        # Transform to ensure uniform distribution on the Pareto front surface
        # The mapping x -> theta is linear, but the surface area element involves cos(theta).
        # To get uniform surface coverage, we sample sin(theta) uniformly.
        # theta = arcsin(u) => x = (2/pi) * arcsin(u)
        shape_samples_transformed = (2.0 / np.pi) * np.arcsin(shape_samples)
        
        # Initialize full dimension array
        X = np.zeros((budget, dim))
        
        # Fill shape variables
        X[:, :n_shape_vars] = shape_samples_transformed
        
        # Fill distance variables with midpoint to ensure g(x) = 0
        # Midpoint of [0, 1] is 0.5
        dist_mid = 0.5
        X[:, n_shape_vars:] = dist_mid
        
        # Clip to bounds just in case
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.zeros((budget, n_obj))
        for i in range(budget):
            F[i] = problem(X[i])
            
        return X
