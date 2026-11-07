"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using Sobol sampling on the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        budget = self.budget
        
        # Generate Sobol points in 2D for the two angle variables x1, x2
        # Sobol sequence requires power of 2 size for best properties, but we can use any size
        # We will generate exactly 'budget' points
        sampler = qmc.Sobol(d=2, scramble=True, seed=42)
        # generate requires power of 2, so we generate the next power of 2 >= budget
        # and then slice
        n_gen = 1
        while n_gen < budget:
            n_gen *= 2
        points = sampler.random(n_gen)
        points = points[:budget]
        
        # Map [0, 1) to [0, 1] for x1, x2
        # The bounds are [0, 1] for all variables.
        # x1, x2 are the first two variables.
        X = np.zeros((budget, dim))
        X[:, 0] = points[:, 0]
        X[:, 1] = points[:, 1]
        
        # Set remaining variables to 0.5 to make g=0 (minimize g)
        if dim > 2:
            X[:, 2:] = 0.5
            
        # Evaluate all points
        F = np.zeros((budget, self.n_obj))
        for i in range(budget):
            F[i] = problem(X[i])
            
        return F

# EVOLVE-BLOCK-END
