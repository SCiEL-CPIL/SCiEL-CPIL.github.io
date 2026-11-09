"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Deterministic Sobol-based global search utilizing the full budget for maximal coverage.
    
    Strategy:
    1. The problem states the minimizer lies in [-4, 4]^5.
    2. We use a scrambled Sobol sequence (deterministic via fixed seed) for space-filling properties.
    3. We map ALL points to [-4, 4]^5 to maximize density in the region of interest.
    4. We evaluate the first 500 points. This avoids the risk of local trapping in a sub-optimal
       basin found during a split-phase search, ensuring consistent global progress.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        from scipy.stats import qmc
        
        best_x = None
        best_f = float('inf')
        
        # Generate Sobol sequence for the full budget
        # Using scramble=True with a fixed seed ensures deterministic but well-distributed points
        # that avoid structural alignment with the hidden function's features.
        sobol_gen = qmc.Sobol(d=dim, scramble=True, seed=42)
        points_unit = sobol_gen.random(budget)
        
        # Map to [-4, 4] to concentrate search density where the minimizer is guaranteed to be.
        # This increases the effective resolution by (10/8)^dim compared to [-5, 5].
        points = points_unit * 8.0 - 4.0
        
        for i in range(budget):
            x = points[i]
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
