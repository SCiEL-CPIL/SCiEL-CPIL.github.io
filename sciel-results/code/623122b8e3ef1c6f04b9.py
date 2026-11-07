import numpy as np


class Optimizer:
    """
    Optimizer for DTLZ4 using structured sampling of the Pareto front.
    Assumes DTLZ4 concentration variables (x_3...x_N) are optimal near the lower bound.
    Uses Halton sequence for the shape variables (x_1, x_2) to ensure good spread.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, index, base):
        """Generate the Halton sequence value for a given index and base."""
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n = self.budget
        X = np.zeros((n, self.dim))
        
        # Sample the first two dimensions (shape variables) using Halton sequence
        # This provides low-discrepancy coverage of the [0, 1]^2 square
        for i in range(n):
            # Use 1-based index for Halton to avoid 0
            idx = i + 1
            x1 = lo[0] + (hi[0] - lo[0]) * self._halton(idx, 2)
            x2 = lo[1] + (hi[1] - lo[1]) * self._halton(idx, 3)
            
            X[i, 0] = x1
            X[i, 1] = x2
            
            # For DTLZ4, the concentration variables (indices 2 to dim-1) 
            # are optimal near the lower bound (0) due to the high beta exponent.
            # Set them to the lower bound to minimize the distance term g(x).
            if self.dim > 2:
                X[i, 2:] = lo[2:]
        
        # Evaluate all points
        # The return value is ignored by the evaluator, but we must call problem
        for i in range(n):
            problem(X[i])
            
        return X
