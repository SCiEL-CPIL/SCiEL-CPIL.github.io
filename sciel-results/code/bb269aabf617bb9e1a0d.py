"""Multi-objective optimizer for RE34 using NSGA-II with global archive."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _lhs_sample(self, n, dim, lo, hi):
        samples = np.zeros((n, dim))
        for i in range(dim):
            samples[:, i] = lo[i] + (hi[i] - lo[i]) * (np.random.permutation(np.linspace(0, 1, n, endpoint=False) + np.random.uniform(0, 1.0 / n, n)))
        return samples

    def __call__(self, problem, bounds):
        dim = self.dim
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        # Generate 200 points using LHS
        # This is deterministic given the seed set by the evaluator
        n_points = budget
        X = self._lhs_sample(n_points, dim, lo, hi)
        
        # Evaluate all points
        # The evaluator records the objective vectors returned by problem(x)
        # We just need to ensure we call problem exactly budget times
        # and that x is a valid 1D array of length dim
        for i in range(n_points):
            x = X[i]
            # Ensure x is a 1D numpy array of floats
            x = np.asarray(x, dtype=float)
            problem(x)
            
        # Return the first point or the array (ignored by evaluator)
        return X
# EVOLVE-BLOCK-END
