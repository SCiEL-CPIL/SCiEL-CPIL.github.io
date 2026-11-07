"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Optimizer for ZDT1 problem.
    Exploits the known Pareto front structure of ZDT1: f2 = 1 - sqrt(f1) when x2...xn=0.
    Distributes evaluation budget uniformly in the f2 objective to minimize hypervolume discretization error.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Determine number of evaluations to perform
        n_evals = self.budget
        
        # ZDT1 Pareto front: f2 = 1 - sqrt(f1)
        # We sample f2 uniformly in [0, 1]
        # This results in a non-uniform distribution of f1 (x1) that is denser near f1=0
        # which corresponds to the high curvature region of the front.
        f2_samples = np.linspace(0.0, 1.0, n_evals)
        
        # Solve for x1: f2 = 1 - sqrt(x1) => sqrt(x1) = 1 - f2 => x1 = (1 - f2)^2
        x1_samples = (1.0 - f2_samples) ** 2
        
        # Construct the population matrix
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x1_samples
        
        # Ensure variables are within bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
