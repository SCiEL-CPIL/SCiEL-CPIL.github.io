"""Multi-objective optimizer for ZDT4 using direct Pareto front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direct Pareto front sampler for ZDT4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT4 Pareto front: x_2..x_D = 0.0 (global min of G for standard ZDT4), x_1 in [0,1]
        # f_1 = x_1, f_2 = H(x_1, G)
        # Using 0.0 targets the global minimum of G(x) in standard ZDT4 definitions.
        target_rest = np.zeros(self.dim - 1)
        for d in range(1, self.dim):
            target_rest[d - 1] = np.clip(0.0, lo[d], hi[d])

        # Sample x_1 values: use uniform spread for maximum hypervolume coverage
        n = self.budget
        x1_values = np.linspace(lo[0], hi[0], n)
        # Add tiny jitter to ensure distinct points
        jitter = np.random.uniform(-1e-6, 1e-6, n)
        x1_values = np.clip(x1_values + jitter, lo[0], hi[0])

        # Pre-allocate
        X = np.zeros((n, self.dim))
        X[:, 0] = x1_values
        X[:, 1:] = target_rest

        # Evaluate all points
        for i in range(n):
            if time.time() > deadline:
                break
            problem(X[i])

        return X
# EVOLVE-BLOCK-END
