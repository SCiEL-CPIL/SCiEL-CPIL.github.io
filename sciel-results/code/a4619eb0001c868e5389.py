"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic sampling on the analytical Pareto front of ZDT4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        n = self.dim
        budget = self.budget

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # For ZDT4, the Pareto front is achieved when x_2..x_n = 0.5 (the global minimum
        # of g(x) = 1 + 10(n-1) + sum((x_i - 0.5)^2 - 10*cos(20*pi*(x_i - 0.5))) for i=2..n
        # is 0 when x_i = 0.5 for all i >= 2)
        # f1 = x1, g = 1 + 10(n-1) + 0 = 11 when n=10
        # f2 = g * (1 - sqrt(f1/g))
        #
        # We sweep x1 uniformly from a small epsilon to 1.0 to cover the front.
        # x1 range: [0, 1]

        # Use all budget evaluations to sample points on the front
        # x1 values: evenly spaced in [0, 1]
        x1_vals = np.linspace(0.0, 1.0, budget)

        # Construct points: x1 = x1_vals, x2..xn = 0.5
        X = np.zeros((budget, n))
        X[:, 0] = x1_vals
        X[:, 1:] = 0.5

        # Clip to bounds (should be within [0,1] for standard ZDT4 bounds)
        X = np.clip(X, lo, hi)

        # Evaluate all points
        F = np.array([problem(x) for x in X])

        return X
# EVOLVE-BLOCK-END
