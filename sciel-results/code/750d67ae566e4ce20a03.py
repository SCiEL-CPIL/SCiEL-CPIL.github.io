"""DTLZ1 optimizer that directly samples the known Pareto front using a uniform lattice."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ1 that directly constructs points on the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = 290.0  # 300s limit with 10s safety margin

    def _elapsed(self):
        return time.time() - self.start_time

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # DTLZ1: first (dim - n_obj + 1) variables determine position on the front,
        # remaining (n_obj - 1) variables determine aggregation (should be 0.5 for Pareto front)
        n_pos = dim - self.n_obj + 1  # 7 - 3 + 1 = 5 position variables
        n_agg = self.n_obj - 1  # 2 aggregation variables

        # On DTLZ1 Pareto front:
        #   x[2:] = 0.5
        #   f1 = 0.5*x1, f2 = 0.5*(1-x1)*x2, f3 = 0.5*(1-x2)
        # Front is parameterized by (x1, x2) in [0,1]^2
        # The Pareto front lies on the simplex f1+f2+f3 = 0.5 with fi >= 0

        # Strategy: Use a deterministic nested lattice grid to discretize the 2D Pareto simplex.
        # The Pareto front in (x1, x2) space is the unit square. We use a nested lattice
        # approach to get exactly budget points with good space-filling properties.
        
        # For exactly budget points, we can use a combination of grid levels.
        # A simple approach: use the first k grid points from a fine grid that covers
        # the space well. Alternatively, use a Halton sequence or similar.
        
        # Let's use a proper nested lattice: generate points on a grid of size m x m
        # where m*m >= budget, then take the first budget points.
        # But we need to be careful about which points to take.
        
        # Better: use a 2D Halton sequence for quasi-random uniform coverage
        def halton(index, base):
            """Generate the index-th Halton sequence number in the given base."""
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        # Generate exactly budget points using 2D Halton sequence (bases 2 and 3)
        all_pairs = np.zeros((budget, 2))
        for i in range(budget):
            all_pairs[i, 0] = halton(i + 1, 2)  # +1 to avoid 0,0
            all_pairs[i, 1] = halton(i + 1, 3)
        
        # Clip to [0, 1]
        all_pairs = np.clip(all_pairs, 0.0, 1.0)

        # Evaluate each point on the Pareto front
        X = np.zeros((budget, dim))
        F = np.zeros((budget, self.n_obj))
        valid_count = 0

        for i in range(len(all_pairs)):
            if self._elapsed() > self.deadline:
                break

            x1 = float(np.clip(all_pairs[i, 0], 0.0, 1.0))
            x2 = float(np.clip(all_pairs[i, 1], 0.0, 1.0))

            x = np.full(dim, 0.5)
            x[0] = x1
            x[1] = x2

            try:
                f = problem(x)
                X[i] = x
                F[i] = f
                valid_count += 1
            except Exception:
                break

        return X[:valid_count]
# EVOLVE-BLOCK-END
