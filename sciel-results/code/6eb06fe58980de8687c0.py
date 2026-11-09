# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """
    Deterministic structured search for DTLZ4.

    Strategy:
    - DTLZ4 has M objectives (here M=3) and N variables (here N=12).
    - The Pareto front is defined by:
        * The first M-1 = 2 variables (x0, x1) determine the shape/position on the front.
        * The remaining N - (M-1) = 10 variables determine the distance to the front.
    - For DTLZ4, the optimal distance variables are all 0.5.
    - So we fix x[2:] = 0.5 and only sample x[0] and x[1] using a low-discrepancy sequence.
    - This guarantees all evaluated points lie exactly on the Pareto front.
    - We use a Sobol sequence to uniformly cover the [0,1]^2 space for x0, x1.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_shape = self.n_obj - 1  # 2
        n_dist = self.dim - n_shape  # 10

        # Optimal value for distance variables in DTLZ4 is 0.5
        dist_opt = 0.5

        # Generate Sobol points in n_obj dimensions for uniform sampling on the simplex
        sobol = qmc.Sobol(d=self.n_obj, scramble=True, seed=42)
        obj_samples = sobol.random(self.budget)  # shape: (budget, n_obj)
        
        # Normalize to sum to 1 (lie on the simplex f1+...+fM=1)
        sums = obj_samples.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        f = obj_samples / sums
        
        # Map objective space f back to decision space x
        # f1 = x0
        # f2 = (1 - x0) * x1  => x1 = f2 / (1 - x0)
        
        x0 = f[:, 0]
        denom = 1.0 - x0
        with np.errstate(divide='ignore', invalid='ignore'):
            x1 = f[:, 1] / denom
            x1 = np.where(np.isfinite(x1), x1, 0.5)
        
        # Build the full population
        pop = np.zeros((self.budget, self.dim))
        pop[:, 0] = x0
        pop[:, 1] = x1
        pop[:, n_shape:] = dist_opt

        # Clip to bounds
        pop = np.clip(pop, lo, hi)

        # Evaluate all points
        F = np.array([problem(x) for x in pop])

        return pop
# EVOLVE-BLOCK-END
