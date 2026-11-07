"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """Deterministic Global Optimization using Sobol sequence."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Use the entire budget for Sobol sampling
        # This ensures maximum uniform coverage and leverages low-discrepancy properties
        # for rapid global exploration, which is crucial for AOCC scoring.
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        
        # Generate 'budget' points in [0, 1)^dim
        sobol_samples = sampler.random(budget)
        
        # Map to [-5, 5]^dim
        mapped_samples = sobol_samples * 10.0 - 5.0

        best_x = None
        best_f = np.inf

        # Evaluate all points in the deterministic sequence
        for i in range(budget):
            x = mapped_samples[i]
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
