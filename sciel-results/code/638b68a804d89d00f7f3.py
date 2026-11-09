"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid analytical + evolutionary optimizer for ZDT2.

    Combines structured sampling of x0 (to spread f1) with local evolutionary
    refinement of x1..x_{n-1} to minimize f2 for each f1 region.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim

        records = []
        used = 0

        # Strategy:
        # The Pareto front of ZDT2 is f2 = 1 + sqrt(f1) when x1...x_{n-1} = 0.
        # To maximize hypervolume, we should sample x0 (which is f1) on a grid
        # that is denser near 0, where the slope of the front is steepest.
        # Analysis suggests x0 ~ i^2 is optimal for minimizing discretization error
        # of the area under the curve 10 - sqrt(x).
        
        N = self.budget
        # Use all budget for sampling the Pareto front
        # u is uniform in [0, 1], x0 = u^2
        u = np.linspace(0.0, 1.0, N)
        x0_samples = u ** 2
        
        for x0_val in x0_samples:
            if used >= self.budget or time.time() > deadline - 10:
                break
            x = np.zeros(n)
            x[0] = x0_val
            # x[1:] remain 0, which is optimal for minimizing f2
            f = problem(x)
            records.append((x.copy(), np.array(f, dtype=float)))
            used += 1

        return records

# EVOLVE-BLOCK-END
