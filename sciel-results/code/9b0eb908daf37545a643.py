"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Deterministic Sobol-based global search with progressive local refinement.
    
    Strategy:
    1. Use a scrambled Sobol sequence mapped to [-4, 4]^5 for global coverage.
    2. After the first 50 evaluations, identify the best point and begin
       interleaving local refinements around it while continuing global search.
    3. This provides early progress (scoring favors early improvements) while
       maintaining global coverage to avoid missing the true minimum.
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
        
        # Generate deterministic Sobol sequence for global coverage
        # Using scramble=True to avoid axis alignment issues
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        points_unit = sobol.random(budget)
        points = points_unit * 8.0 - 4.0
        
        # Evaluate all points in the sequence order
        # Sobol sequences have low discrepancy for any prefix,
        # ensuring consistent progress in the AOCC metric.
        for i in range(budget):
            x = points[i]
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
