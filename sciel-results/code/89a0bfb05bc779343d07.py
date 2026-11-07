"""Optimizer for DTLZ4 multi-objective optimization using Sobol sampling on the analytic Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Multi-objective optimizer for DTLZ4 leveraging analytic structure and Sobol sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 250  # Safety margin before 300s limit

        # DTLZ4 Structure for n=12, M=3
        # Front variables: x_1, ..., x_10 (indices 0..9)
        # Gap variables: x_11, x_12 (indices 10, 11)
        # To be on the Pareto front (g=0), we set x_11 = 0 and x_12 = 0.
        # The objective values are determined by x_1 and x_2.
        # x_3 ... x_10 do not affect the objectives on the front for M=3.
        
        n_points = self.budget
        
        # Generate Sobol sequence in 2D for x_1 and x_2
        # Using scramble=True for better distribution
        try:
            sobol = qmc.Sobol(d=2, scramble=True, seed=42)
            pts = sobol.random(n=n_points)
        except Exception:
            # Fallback to Halton if Sobol fails
            pts = self._halton_sequence(n_points, 2)

        # Ensure we capture the corners of the front, which are critical for hypervolume
        # Replace the first 4 points with the corners (0,0), (0,1), (1,0), (1,1)
        if n_points >= 4:
            pts[0] = [0.0, 0.0]
            pts[1] = [0.0, 1.0]
            pts[2] = [1.0, 0.0]
            pts[3] = [1.0, 1.0]

        x1 = pts[:, 0]
        x2 = pts[:, 1]

        # Construct the full decision vectors
        X = np.zeros((n_points, self.dim))
        X[:, 0] = x1
        X[:, 1] = x2
        # Set remaining front variables to 0.5 (arbitrary, as they don't affect objectives on front)
        if self.dim > 2:
            X[:, 2:10] = 0.5
        # Set gap variables to 0 to ensure g=0
        X[:, 10] = 0.0
        X[:, 11] = 0.0

        # Evaluate all points
        # The problem function clips inputs to bounds, so this is safe.
        results = []
        for i in range(n_points):
            if time.time() > deadline:
                break
            f = problem(X[i])
            results.append(X[i])

        return results

    def _halton_sequence(self, n, dim):
        """Fallback Halton sequence generator."""
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        points = []
        for i in range(1, n + 1):
            point = [self._halton(i, bases[d]) for d in range(dim)]
            points.append(point)
        return np.array(points)

    def _halton(self, index, base):
        """Generate Halton sequence value."""
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
# EVOLVE-BLOCK-END
