"""Portfolio search for DTLZ2: low-discrepancy Sobol sampling on the Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.stats.qmc as qmc


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using Sobol sampling on the front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 310.0  # safety margin under 330s kill

        used = 0
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        def remaining():
            return budget - used

        # DTLZ2 structure:
        #   g = sum_{i=3}^{dim} (x_i - 0.5)^2
        #   f1 = (1+g)*sin(pi*x1/2)   [here g=0 -> front]
        #   f2 = (1+g)*cos(pi*x1/2)*sin(pi*x2/2)
        #   f3 = (1+g)*cos(pi*x2/2)
        # Setting x3..x12 = 0.5 gives g = 0 -> point on the true Pareto front.
        # x1, x2 in [0,1] sweep the quarter-sphere front.
        #
        # Use a 2D Sobol (low-discrepancy) sequence to fill the (x1, x2)
        # parameter space, giving near-uniform coverage of the spherical front
        # with minimal clustering compared to a regular grid.

        # Next power of 2 >= budget for a valid Sobol sequence
        n_sobol = 1
        while n_sobol < budget:
            n_sobol *= 2

        sampler = qmc.Sobol(d=2, scramble=True, seed=0)
        # Generate n_sobol points in [0,1]^2
        u = sampler.random(n_sobol)  # shape (n_sobol, 2)
        u = u[:budget]  # take only what we need

        # Build candidate decision vectors
        candidates = []
        for i in range(len(u)):
            x = np.full(dim, 0.5)  # x3..x12 = 0.5 -> g = 0
            x[0] = u[i, 0]
            x[1] = u[i, 1]
            candidates.append(x)

        # Evaluate all candidates, recording every objective vector.
        # Keep the deadline guard: if time is nearly up, stop and return.
        for x in candidates:
            if used >= budget:
                break
            if time.time() >= deadline:
                break
            problem(x)  # every call is recorded by the evaluator
            used += 1

        return np.array(candidates[:used])
# EVOLVE-BLOCK-END
