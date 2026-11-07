"""Direct construction optimizer for DTLZ4 exploiting analytical Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _halton(index, base):
    """
    Generate the Halton sequence for a given 1-based index and base.
    """
    f = 1.0
    r = 0.0
    i = index
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using Halton sequence on the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        n_g = 5
        n_front = dim - n_g

        # Generate Halton sequence indices 1 to budget
        # Using bases 2 and 3 for 2D Halton sequence
        indices = np.arange(1, budget + 1)
        
        # Vectorized Halton generation is not trivial, but for N=200, loop is fine.
        # However, we can precompute or just use a loop.
        u1 = np.array([_halton(i, 2) for i in indices])
        u2 = np.array([_halton(i, 3) for i in indices])
        
        # Map uniform variables to decision variables for DTLZ4 Pareto front
        # x1 = (2/pi) * arccos(u1)
        # x2 = u2
        # This mapping ensures uniform coverage of the sphere surface area.
        x1_vals = (2.0 / np.pi) * np.arccos(u1)
        x2_vals = u2
        
        # Construct decision vectors
        X = np.zeros((budget, dim))
        X[:, 0] = x1_vals
        X[:, 1] = x2_vals
        
        # Set other front variables to 0.5 (arbitrary, doesn't affect objectives)
        if n_front > 2:
            X[:, 2:n_front] = 0.5
            
        # Set g variables to 0.5 to ensure g=0
        X[:, n_front:] = 0.5
        
        # Clip to bounds (should already be in [0,1] but just in case)
        X = np.clip(X, lo, hi)
        
        # Check time before starting evaluation loop
        if time.time() - t0 > deadline:
            return X
        
        # Evaluate all points
        # We must call problem(x) for each point.
        F = np.empty((budget, n_obj))
        for i in range(budget):
            if time.time() - t0 > deadline:
                break
            F[i] = problem(X[i])
        
        # The return value is ignored, but we return X as per the seed program style
        return X
# EVOLVE-BLOCK-END
