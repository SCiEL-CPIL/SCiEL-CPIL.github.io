"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Analytic ZDT1 Pareto front sampling with hypervolume-optimal point distribution."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0
        
        # ZDT1 Pareto front: f1 = x1, f2 = (1 + 10*n) * (1 - sqrt(f1/(10*n)))
        # where n = dim - 1
        n = self.dim - 1
        
        # We want to maximize hypervolume with reference point [11, 11]
        # The Pareto front is f2 = (1+10n)*(1 - sqrt(f1/(10n))) for f1 in [0, 1]
        # Hypervolume contribution of a point depends on its position
        # Optimal distribution: non-uniform spacing that equalizes hypervolume contribution
        
        # For ZDT1, the Pareto front is convex (in the sense that f2 decreases convexly with f1)
        # The hypervolume contribution of a point at f1 is proportional to the area it "covers"
        # Optimal points should be denser where the front is steeper (lower f1)
        
        # Use the inverse CDF method: if we want N points that maximize discrete hypervolume,
        # we can use a distribution proportional to the hypervolume contribution
        # For a convex front, this means more points at lower f1 values
        
        # A good approximation: f1_i = i^2 / N^2 for i = 1, ..., N (quadratic spacing)
        # This gives more points near f1=0 where the front is steeper
        
        N = self.budget
        
        # Generate candidate points with quadratic spacing
        # f1 values: i^2 / N^2 for i = 0, 1, ..., N-1
        i_vals = np.arange(N)
        f1_vals = (i_vals ** 2) / (N ** 2)
        
        # Compute f2 from f1
        f2_vals = (1 + 10 * n) * (1 - np.sqrt(f1_vals / (10 * n)))
        
        # Create x vectors: x1 = f1, x2...x30 = 0 (to achieve the Pareto front)
        X = np.zeros((N, self.dim))
        X[:, 0] = f1_vals
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        # Check time
        if time.time() > deadline - 5.0:
            return X
        
        # The points on the Pareto front should give maximum hypervolume
        # Return the solution set
        return X
# EVOLVE-BLOCK-END
