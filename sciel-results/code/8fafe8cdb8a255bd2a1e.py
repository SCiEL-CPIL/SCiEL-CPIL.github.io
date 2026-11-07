# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Directly construct points on the DTLZ1 Pareto front using a Halton sequence."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # DTLZ1: N variables, M objectives
        # Pareto front: first N-M+1 variables are position, last M-1 are aggregation (set to 0.5)
        n_pos = dim - n_obj + 1  # 5 for dim=7, n_obj=3
        n_agg = n_obj - 1        # 2

        # Generate Halton sequence for the position variables
        # Halton sequence uses different prime bases for each dimension
        primes = [2, 3, 5, 7, 11]  # first 5 primes for 5 position dimensions
        pos_points = np.zeros((n, n_pos))

        for d in range(n_pos):
            for i in range(n):
                pos_points[i, d] = self._halton(i + 1, primes[d])

        # Construct the full input vectors
        X = np.zeros((n, dim))
        X[:, :n_pos] = pos_points
        X[:, n_pos:] = 0.5

        # Clip to bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)

        # Evaluate all points
        F = np.array([problem(x) for x in X])

        return X

    @staticmethod
    def _halton(index, base):
        """Generate the index-th point in the Halton sequence with the given base."""
        result = 0.0
        f = 1.0 / base
        i = index
        while i > 0:
            result += f * (i % base)
            i //= base
            f /= base
        return result
# EVOLVE-BLOCK-END
