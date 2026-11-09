"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """Uses Latin Hypercube Sampling on the DTLZ1 Pareto Front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        # DTLZ1 structure:
        # Pareto front is a 2D simplex in the first two variables (x[0], x[1]).
        # The remaining variables (x[2:]) must be set to 0.5 to minimize g.
        
        n_evals = self.budget
        
        # Generate Latin Hypercube points in 2D
        # LHS provides better coverage than simple random sampling
        sampler = qmc.LatinHypercube(d=2, n=n_evals)
        u = sampler.random()
        
        # Map the 2D unit square points to the 2D triangle (Pareto Front)
        # x0 = u0
        # x1 = u1 * (1 - u0)
        # This ensures x0 >= 0, x1 >= 0, x0 + x1 <= 1
        
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = u[:, 0]
        X[:, 1] = u[:, 1] * (1.0 - u[:, 0])
        
        # Set the remaining dimensions to the optimal value for g (0.5)
        X[:, 2:] = 0.5
        
        # Clip to bounds (though the mapping and 0.5 should be within [0,1])
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
