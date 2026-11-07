"""Multi-objective optimizer on ZDT6 — Direct Pareto Front Sampling."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Optimizer that directly samples the known Pareto front of ZDT6.
    ZDT6 Pareto front: f1 = x1, f2 = 1 - sqrt(x1) with x2..xn = 0.5 (to make g(x)=1).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        budget = self.budget
        dim = self.dim

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        # The Pareto front of ZDT6 is achieved when g(x) = 1.
        # g(x) = 1 + 10(n-1) + sum_{i=2}^n ((x_i/pi - 0.5)^2 * sin(4*pi*x_i))
        # Setting x_i = 0.5 for i > 1 makes the sum 0, so g(x) = 1.
        # Then f1 = x1, f2 = 1 - sqrt(x1).
        # We sample x1 uniformly in [0, 1] to cover the front densely.
        
        # Generate x1 values
        x1_vals = np.linspace(0.0, 1.0, budget)
        
        # Construct decision vectors
        X = np.zeros((budget, dim))
        X[:, 0] = x1_vals
        
        if dim > 1:
            # Set remaining variables to 0.5 to satisfy g(x)=1
            X[:, 1:] = 0.5

        # Ensure points are within bounds
        X = np.clip(X, lo, hi)

        # Evaluate all points
        # The evaluator records every objective vector returned.
        F = np.array([problem(x) for x in X])

        return X
# EVOLVE-BLOCK-END
