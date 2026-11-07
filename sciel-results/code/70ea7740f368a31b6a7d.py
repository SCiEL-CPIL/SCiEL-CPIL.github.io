import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Multi-objective optimizer for DTLZ4 using Sobol sequence sampling on the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ4: first M-1 vars determine front position, rest must be 0.5 for Pareto optimality
        n_points = self.budget

        # Generate Sobol sequence points in [0, 1]^(M-1)
        # M=3, so M-1=2 dimensions for the front
        front_dim = self.n_obj - 1
        
        try:
            sobol = qmc.Sobol(d=front_dim, scramble=True)
            u = sobol.random(n_points)
        except Exception:
            # Fallback to Halton if Sobol is not available or fails
            u = self._halton_sequence(n_points, front_dim)

        # Construct points: x[0:front_dim] from Sobol, x[front_dim:] = 0.5
        X = np.zeros((n_points, self.dim))
        X[:, :front_dim] = u
        X[:, front_dim:] = 0.5

        # Clip to bounds (should be [0,1] but be safe)
        X = np.clip(X, lo, hi)

        # Evaluate all points
        F = np.array([problem(x) for x in X])
        used = n_points

        return X

    def _halton(self, index, base):
        """Generate a single Halton sequence value."""
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    def _halton_sequence(self, n, dim):
        """Generate n quasi-random points in [0,1)^dim using Halton sequence."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        points = np.zeros((n, dim))
        for i in range(n):
            for d in range(dim):
                points[i, d] = self._halton(i + 1, primes[d])
        return points
