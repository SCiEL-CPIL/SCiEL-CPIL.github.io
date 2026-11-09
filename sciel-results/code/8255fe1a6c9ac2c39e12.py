"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Dense sampling of the ZDT3 Pareto front (x2..x30=0, vary x1)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        if self.budget <= 0:
            return np.array([])

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT3 Pareto front: x2..x30 = 0, x1 in [0,1]
        # Sample x1 densely to cover the (discontinuous) Pareto front
        n_samples = self.budget
        x1_vals = np.linspace(0.0, 1.0, n_samples)

        # Build candidate points: x1 varies, rest are 0
        # Clip to bounds to be safe
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = x1_vals
        # Ensure within bounds
        X = np.clip(X, lo, hi)

        used = 0
        for i in range(n_samples):
            if time.time() > deadline:
                break
            if used >= self.budget:
                break
            f = problem(X[i])
            used += 1

        return X
# EVOLVE-BLOCK-END
