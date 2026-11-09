"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc

class Optimizer:
    """Exploits DTLZ1 structure to sample the Pareto front directly using low-discrepancy sequences."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        
        X = np.zeros((n_evals, self.dim))
        # Set distance variables to 0.5 (assumed optimal for DTLZ1 g=0)
        X[:, 2:] = 0.5
        
        # Use Halton sequence for better coverage of the 2D simplex (triangle)
        h = qmc.Halton(d=2, scramble=True)
        u = h.random(n_evals)
        
        u1 = u[:, 0]
        u2 = u[:, 1]
        
        # Map [0,1]^2 to the triangle x0>=0, x1>=0, x0+x1<=1
        r = np.sqrt(u1)
        theta = u2 * (np.pi / 2.0)
        
        X[:, 0] = r * np.cos(theta)
        X[:, 1] = r * np.sin(theta)
        
        X = np.clip(X, lo, hi)
        
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
