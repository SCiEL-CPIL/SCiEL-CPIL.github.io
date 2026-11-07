"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using LHS-based exploration for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        dim = self.dim
        budget = self.budget
        
        # Use all budget for Latin Hypercube Sampling
        # LHS ensures uniform coverage in each dimension
        n = budget
        rng = np.random.default_rng(0)
        
        # Generate LHS points
        # For each dimension, divide [0,1] into n equal intervals and pick one random point in each
        lhs = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            lhs[:, d] = (perm + rng.random(n)) / n
        
        # Map to bounds
        X = lo + range_ * lhs
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
