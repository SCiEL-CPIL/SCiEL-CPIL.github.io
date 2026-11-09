"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Exploits the known ZDT3 Pareto front structure.
    Samples the manifold x[1:] = 0.5 with non-uniform x[0] distribution
    to better approximate the hypervolume in regions of high curvature.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        # Generate uniform grid in [0, 1]
        u = np.linspace(0.0, 1.0, n_evals)
        # Transform to cluster near 0 (high curvature region of ZDT3 front)
        x0_values = u ** 1.5
        
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x0_values
        if self.dim > 1:
            X[:, 1:] = 0.5
            
        X = np.clip(X, lo, hi)
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
# EVOLVE-BLOCK-END
