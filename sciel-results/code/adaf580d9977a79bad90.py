"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using Latin Hypercube Sampling for uniform coverage."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 320.0 # Safety margin

        dim = self.dim
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Generate Latin Hypercube Sampling points
        # For each dimension, divide [0, 1] into 'budget' bins and pick one random point in each bin.
        # To make it deterministic and reproducible without relying on external RNG state 
        # (since np.random.seed is called by evaluator), we use a fixed seed for our internal RNG
        # or just use np.random which is seeded by the evaluator.
        # The evaluator calls np.random.seed(seed) before creating Optimizer.
        # So np.random calls will be deterministic per seed.
        
        # LHS Generation
        X_lhs = np.zeros((budget, dim))
        for j in range(dim):
            # Create bins
            indices = np.arange(budget)
            # Random permutation for this dimension
            perm = np.random.permutation(budget)
            # Uniform random offsets within bins
            offsets = np.random.rand(budget)
            # Position in [0, 1]
            u = (perm + offsets) / budget
            X_lhs[:, j] = u
        
        # Map to bounds
        X = lo + X_lhs * span
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        # Check time, though 200 evals should be fast
        if time.time() > deadline:
            return X

        return X
# EVOLVE-BLOCK-END
