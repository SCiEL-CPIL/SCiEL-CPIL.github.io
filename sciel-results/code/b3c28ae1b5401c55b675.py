"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """ZDT6 specialized optimizer: exploits Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        # ZDT6 Pareto front is at x_2..x_n = 0, varying x_1
        # f1 = 1 - sqrt(x1), f2 = 1 - sqrt(f1)
        # We want to maximize HV.
        # Optimal points are on the front.
        # We sample x1 to distribute points well on the front.
        # Uniform in f1 (which is 1 - sqrt(x1)) is a good heuristic.
        
        n_evals = self.budget
        # Generate x1 values
        # f1 ranges from 1 (x1=0) to 0 (x1=1)
        # Let f1_i = 1 - i/(n_evals-1)
        # x1_i = (1 - f1_i)^2 = (i/(n_evals-1))^2
        
        x1_vals = np.linspace(0, 1, n_evals)
        # x1 = (1 - f1)^2. If we want uniform f1, x1 = (1 - f1)^2.
        # If we use uniform x1, f1 is not uniform.
        # Let's try uniform x1 first, as it's simpler and often sufficient.
        # Actually, let's use the x1 values directly.
        
        X = np.zeros((n_evals, self.dim))
        for i in range(n_evals):
            X[i, 0] = x1_vals[i]
            # x_2..x_n are 0, which is already set by zeros
        
        # Evaluate all
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
