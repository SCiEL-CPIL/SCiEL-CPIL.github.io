"""Deterministic LHS-based multi-objective optimizer for WFG9."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """Latin Hypercube Sampling based optimizer for WFG9 (3 objectives, 12 dims)."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        n = self.budget
        dim = self.dim
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Generate Latin Hypercube samples
        sampler = qmc.LatinHypercube(d=dim, seed=42)
        samples_unit = sampler.random(n=n)

        # Map to bounds
        X = lo + (hi - lo) * samples_unit

        # Evaluate all points
        F = np.array([problem(x) for x in X])

        return X
# EVOLVE-BLOCK-END
