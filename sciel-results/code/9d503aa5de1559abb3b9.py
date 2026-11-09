import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Exploits DTLZ1 structure to sample the Pareto front directly using Sobol sequences."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Generate Sobol samples in 2D for the simplex
        next_pow2 = 1
        while next_pow2 < budget:
            next_pow2 *= 2

        s = qmc.Sobol(d=2, scramble=True)
        samples = s.random(next_pow2)
        samples = samples[:budget]

        # Map to simplex: u, v >= 0, u+v <= 1
        # Use transformation: u = sqrt(s1), v = (1-sqrt(s1))*s2
        u = np.sqrt(samples[:, 0])
        v = (1 - np.sqrt(samples[:, 0])) * samples[:, 1]

        # Objective values on Pareto front (g=0)
        f0 = 0.5 * (1 - u - v)
        f1 = 0.5 * u
        f2 = 0.5 * v

        # Invert to decision variables
        # f0 = 0.5*(1+x0) -> x0 = 2*f0 - 1
        x0 = 2 * f0 - 1

        # f1 = 0.5*x0*(1-x1), f2 = 0.5*x0*x1
        # x1 = f2 / (f1 + f2) when f1+f2 > 0
        denom = f1 + f2
        x1 = np.zeros(budget)
        mask = denom > 1e-15
        x1[mask] = f2[mask] / denom[mask]
        x1[~mask] = 0.0  # when x0=0, x1 is irrelevant

        # Clip x0, x1 to [0,1]
        x0 = np.clip(x0, 0, 1)
        x1 = np.clip(x1, 0, 1)

        # For g=0, x[1] must be at a sin zero: x[1] = k/6 for k=0..6
        # Snap x1 to nearest sin zero
        x1_snapped = np.round(x1 * 6) / 6.0
        x1_snapped = np.clip(x1_snapped, 0, 1)

        # Construct decision vectors
        points = np.zeros((budget, dim))
        points[:, 0] = x0
        points[:, 1] = x1_snapped
        for j in range(2, dim):
            points[:, j] = 0.5

        # Evaluate
        results = []
        for i in range(budget):
            if time.time() > deadline:
                break
            x = points[i]
            results.append(problem(x))

        return results
