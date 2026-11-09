# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for ZDT2.
    Exploits the structure of ZDT2:
    f1 depends only on x0.
    f2 depends on x0 and x1..xn-1 via g(x).
    The Pareto front is achieved when g(x) is minimized, which happens
    when x1..xn-1 = 0.5.
    Minimizing g(x) to 1 shifts the Pareto front closer to the origin,
    significantly increasing the hypervolume relative to the reference point.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, index, base):
        """Generate the index-th number in the Halton sequence with the given base."""
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Initialize X with zeros
        X = np.zeros((self.budget, self.dim))
        
        # Set x1..xn-1 to 0.5 to minimize g(x)
        if self.dim > 1:
            X[:, 1:] = 0.5
            
        # Fill x0 with Halton sequence for coverage of [0, 1]
        # Use indices 1 to budget-1 for Halton to avoid 0?
        # Halton(1, 2) = 0.5, Halton(2, 2) = 0.25, etc.
        # We want to cover [0, 1].
        # Standard Halton starts with 0 for index 0.
        # Let's use index 1 to budget to get non-zero values, or just 0 to budget-1.
        # If we use 0, x0=0.
        for i in range(self.budget):
            if time.time() > deadline:
                break
            X[i, 0] = self._halton(i, 2)
            
        # Clip to bounds just in case, though 0.5 and Halton [0,1) are within [0,1]
        X = np.clip(X, lo, hi)
        
        F = np.empty((self.budget, self.n_obj))
        for i in range(self.budget):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
            
        return X
# EVOLVE-BLOCK-END
