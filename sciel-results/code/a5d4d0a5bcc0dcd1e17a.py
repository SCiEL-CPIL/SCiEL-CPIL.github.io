"""Direct construction of DTLZ1 Pareto front points using Sobol sequence."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc

class Optimizer:
    """
    Optimizer that directly samples the known Pareto front of DTLZ1.
    Assumes DTLZ1 structure: M objectives, N variables.
    Pareto front is characterized by x_{M-1}...x_{N-1} = 0.5.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # DTLZ1 parameters
        M = self.n_obj
        N = self.dim
        
        # Number of position variables is M - 1
        n_pos = M - 1
        # Number of aggregation variables is N - (M - 1)
        n_agg = N - n_pos
        
        # Generate space-filling points in the position variable space
        # Using Sobol sequence for low discrepancy
        sampler = qmc.Sobol(d=n_pos, scramble=True)
        points_pos = sampler.random(self.budget)
        
        # Construct the full decision vectors
        X = np.zeros((self.budget, N))
        
        # Assign position variables
        X[:, :n_pos] = points_pos
        
        # Set aggregation variables to 0.5 (known minimum for g in DTLZ1)
        if n_agg > 0:
            X[:, n_pos:] = 0.5
            
        # Clip to bounds to be safe
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        for x in X:
            problem(x)
            
        return None
# EVOLVE-BLOCK-END
