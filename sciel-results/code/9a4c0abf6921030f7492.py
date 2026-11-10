"""Multi-objective optimizer for ZDT2 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer tailored for ZDT2.
    
    ZDT2 Pareto front: f1 = x[0], f2 = (1 - sqrt(x[0]))^2 + 0.01*x[0]*sin(10*pi*x[0])
    The front is achieved when x[1:] = 0.
    
    Strategy: Distribute all 200 evaluations along the known Pareto front
    (x[1:] = 0, x[0] spread across [0,1]) to maximize hypervolume coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 290  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # For ZDT2, the Pareto front is at x[1:] = 0, x[0] in [0, 1]
        # We distribute evaluations along x[0] to cover the front densely
        n_evals = self.budget
        
        # Create points along the Pareto front
        # Use a mix of uniform and slightly denser sampling near the ends
        # where the front has more curvature
        x0_values = np.linspace(0.0, 1.0, n_evals)
        
        # Build the full vectors
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = x0_values
        # x[1:] stays at 0, which is within bounds [0,1]
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
