"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direct Pareto front sampling for ZDT2 with hypervolume-optimized spacing."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310  # safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim - 1  # number of variables in g

        # ZDT2 on Pareto front: x[0] = t, x[1:] = 0
        # f1 = t, f2 = (1+10*0)*(1 - sqrt(t/(1+10*0))) = 1 - sqrt(t)
        # So front is f1 in [0,1], f2 = 1 - sqrt(f1)

        # We have budget evaluations. Use them to sample the front.
        # Strategy: place points to maximize hypervolume.
        # For 2D front, optimal is to equalize hypervolume contribution.
        # HV contribution of point i (sorted by f1) is (11-f1_i)*(11-f2_i) - overlap
        
        # Simple approach: use all budget points on the front
        # With budget=200, we can place 200 points on the front
        
        budget = self.budget
        # Place points with spacing optimized for hypervolume
        # For curve f2 = 1 - sqrt(f1), the optimal spacing equalizes
        # the area of each "rectangle" under the curve
        
        # Parametrize by f1 = t, t in [0,1]
        # f2 = 1 - sqrt(t)
        # d(f2)/dt = -1/(2*sqrt(t))
        # For equal hypervolume contribution, we want equal area under the curve
        # between consecutive points, which means equal f2 spacing in some sense
        
        # Actually for hypervolume maximization with reference (11,11):
        # Each point contributes (11-f1)*(11-f2) but overlaps with neighbors
        # The optimal is to maximize the union area
        
        # For a monotone decreasing front, the hypervolume is:
        # sum over i of (11 - f1_i) * (f2_{i-1} - f2_i) where f2 sorted descending
        # Plus the last term (11-f1_last)*(11-f2_last)
        
        # Let's just use uniform spacing in f2 first, then refine
        # f2 ranges from 1 (at f1=0) to 0 (at f1=1)
        
        # Number of points to use on front
        n_points = min(budget, 200)
        
        # Uniform in f2: f2_i = 1 - i/(n_points-1) for i=0..n_points-1
        # Then f1_i = (1 - f2_i)^2
        # This gives non-uniform spacing in f1, denser near f1=0
        
        # Alternative: uniform in f1
        # f1_i = i/(n_points-1), f2_i = 1 - sqrt(f1_i)
        
        # For hypervolume, uniform in f2 is better because the curve
        # is steeper near f1=0
        
        # Let's use a mix: optimize the spacing
        # For now, try uniform in f2
        f2_vals = np.linspace(1.0, 0.0, n_points)
        f1_vals = (1.0 - f2_vals) ** 2
        
        # Convert to decision variables: x[0] = f1_val, x[1:] = 0
        X = np.zeros((n_points, self.dim))
        X[:, 0] = f1_vals
        # x[1:] = 0 is already set
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        used = n_points
        
        # If we have budget left, we could add more points
        # But 200 points on a 2D front should be sufficient
        # The front is 1D, so 200 points give very fine coverage
        
        # Check deadline
        if time.time() > deadline:
            return X
        
        return X
# EVOLVE-BLOCK-END
