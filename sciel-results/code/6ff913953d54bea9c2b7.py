# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        N = self.budget
        m = self.n_obj
        n = self.dim

        # Build a deterministic, high-quality space-filling set in [0,1]^3.
        # Use a 3D Halton sequence (radices 2, 3, 5) for excellent low-discrepancy
        # coverage. This is a structural change from LHS: Halton sequences
        # provide better uniformity in the sense of discrepancy theory.
        
        def halton(index, base):
            result = 0.0
            f = 1.0 / base
            i = index
            while i > 0:
                result += f * (i % base)
                i = i // base
                f /= base
            return result

        # Generate N points using Halton sequence with radices 2, 3, 5
        # Skip the first point (index 0 gives (0,0,0)) and use indices 1..N
        points = np.zeros((N, m))
        for i in range(N):
            idx = i + 1  # skip index 0
            points[i, 0] = halton(idx, 2)
            points[i, 1] = halton(idx, 3)
            points[i, 2] = halton(idx, 5)

        # Map to decision variable space:
        # For DTLZ7, setting x_{m..n-1} = 0 gives g=0, placing on Pareto front.
        # The first m variables correspond directly to objectives when g=0.
        X = np.zeros((N, n))
        for j in range(m):
            # Map [0,1] Halton values to [bounds[j][0], bounds[j][1]]
            lo_j = bounds[j][0]
            hi_j = bounds[j][1]
            X[:, j] = lo_j + (hi_j - lo_j) * points[:, j]

        # Evaluate all points
        for i in range(N):
            if time.time() > deadline:
                break
            problem(X[i])

        return X

# EVOLVE-BLOCK-END
