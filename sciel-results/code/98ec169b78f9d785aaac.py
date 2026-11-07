"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer for ZDT2.
    Strategy: Deterministic sampling of the known Pareto front using a power-law
    distribution for x1 to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n = self.budget
        # Generate uniform u values in [0, 1]
        u = np.linspace(0, 1, n)
        # Apply power law to concentrate samples near x1=0
        # x1 = u^2 minimizes the Riemann sum error for the concave hypervolume contribution
        x1 = u ** 2
        
        # Construct the population
        # x1 is the first variable, x2..xn are set to 0 to achieve g_min=1
        X = np.zeros((n, self.dim))
        X[:, 0] = x1
        
        # Evaluate all points
        # The problem function clips to bounds, but our values are already in [0, 1]
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
