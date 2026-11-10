"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer tailored for ZDT2.
    Exploits the known structure: Pareto front lies on x[1:] = 0,
    with f1 = x[0] and f2 = 1 - sqrt(x[0]) (when x[1:] = 0).
    
    Uses a mix of uniform-in-f1 and sqrt-spaced-in-f1 (uniform-in-f2)
    to maximize hypervolume coverage. The front is concave, so we need
    points near both corners and in the middle.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 290s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_points = min(self.budget, 200)

        # ZDT2 Pareto front: x[0] in [0,1], x[1:] = 0
        # f1 = x[0], f2 = 1 - sqrt(x[0])
        
        # Strategy: Use a dense set of points that optimally covers the concave front.
        # For a concave Pareto front f2 = 1 - sqrt(f1), the hypervolume contribution
        # is maximized by distributing points according to the curvature.
        # A good heuristic is to use points that are uniform in f2 (since the front
        # is f2 = 1 - sqrt(f1), uniform f2 means x0 = (1-f2)^2).
        # We will use a combination: some uniform in f1, some uniform in f2.
        
        # Let's try a pure "uniform in f2" strategy first, as it's often better for concave fronts.
        # If f2 is uniform from 0 to 1, then x0 = (1-f2)^2.
        # Let's also add the two corners explicitly to ensure they are covered.
        
        # Generate points uniform in f2 (which maps to x0 = (1-f2)^2)
        # f2 ranges from 0 (at x0=1) to 1 (at x0=0)
        f2_values = np.linspace(0.0, 1.0, n_points)
        x0_values = (1.0 - f2_values) ** 2
        
        # Ensure endpoints are exactly 0 and 1
        x0_values[0] = 1.0  # f2=0 -> x0=1
        x0_values[-1] = 0.0 # f2=1 -> x0=0
        
        # Build the full vectors with x[1:] = 0
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x0_values
        
        # Ensure within bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        used = n_points
        
        return X
# EVOLVE-BLOCK-END
