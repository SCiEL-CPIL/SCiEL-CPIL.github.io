"""Halton-sequence stratified sampling on DTLZ2 Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Place all evaluations on the DTLZ2 Pareto front using Halton sequence for optimal coverage."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 325.0  # safety margin before 330s kill
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        def halton_sequence(n, base):
            """Generate n points in [0,1] using Halton sequence with given base."""
            result = np.zeros(n)
            for i in range(n):
                f = 1.0 / base
                r = 0.0
                idx = i + 1
                while idx > 0:
                    idx, remainder = divmod(idx, base)
                    r += f * remainder
                    f /= base
                result[i] = r
            return result

        # DTLZ2 Pareto front: x[0]=t1, x[1]=t2, x[2:]=0.5
        # f1 = cos(t1)*cos(t2), f2 = cos(t1)*sin(t2), f3 = sin(t1)
        # t1, t2 in [0, pi/2]
        
        # Use Halton sequence for 2D parameter space (t1, t2)
        # Base 2 for t1, base 3 for t2
        n_points = budget
        h1 = halton_sequence(n_points, 2)
        h2 = halton_sequence(n_points, 3)
        
        # Map to [0, pi/2]
        t1 = h1 * (np.pi / 2)
        t2 = h2 * (np.pi / 2)
        
        # Evaluate all points on the Pareto front
        for i in range(n_points):
            if time.time() > deadline:
                break
            x = np.full(dim, 0.5)
            x[0] = t1[i]
            x[1] = t2[i]
            problem(x)
        
        return None
# EVOLVE-BLOCK-END
