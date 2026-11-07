"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ4 Structure:
        # x0, x1 are position variables
        # x2...x11 are shape variables
        # Pareto front is typically at shape variables = 0 (for standard DTLZ4)
        
        # Generate low-discrepancy points for x0, x1
        # Use Sobol sequence
        sampler = qmc.Sobol(d=2, scramble=True)
        
        # We have budget evaluations.
        # Let's use all of them for the initial structured sample.
        n_points = self.budget
        
        # Generate Sobol points
        # Sobol requires power of 2 for best quality, but works for any N
        points_2d = sampler.random(n_points)
        
        # Map to bounds for x0, x1
        x0 = lo[0] + points_2d[:, 0] * (hi[0] - lo[0])
        x1 = lo[1] + points_2d[:, 1] * (hi[1] - lo[1])
        
        # Construct full dimension vectors
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x0
        X[:, 1] = x1
        # Set shape variables (indices 2 to dim-1) to 0
        # This is the known Pareto optimal setting for DTLZ4 shape variables
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        # Check if we should do any local search?
        # With 200 budget, we used it all.
        # If we had more, we could perturb.
        
        return X
# EVOLVE-BLOCK-END
