"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Sobol-based global exploration + local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # Safety margin for 300s limit
        
        # Phase 1: Sobol-based global exploration
        # Use first 400 points for exploration
        n_explore = min(400, budget)
        n_refine = budget - n_explore
        
        # Generate Sobol points
        sampler = qmc.Sobol(d=dim, scramble=True)
        # Next power of 2 >= n_explore
        m = 1
        while m < n_explore:
            m *= 2
        sobol_points = sampler.random(m)
        
        # Map to [-5, 5]
        sobol_points = sobol_points * 10 - 5
        
        best_x = None
        best_f = float('inf')
        
        # Evaluate exploration points
        for i in range(n_explore):
            if time.time() > deadline:
                break
            x = sobol_points[i]
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Local refinement around the best point found
        if best_x is not None and n_refine > 0:
            # Use a simple local search with decreasing step size
            sigma = 1.0
            for _ in range(n_refine):
                if time.time() > deadline:
                    break
                # Try several directions
                for _ in range(5):  # 5 attempts per refinement step
                    if time.time() > deadline:
                        break
                    y = np.clip(best_x + sigma * np.random.standard_normal(dim), -5, 5)
                    fy = func(y)
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                        sigma *= 1.2
                    else:
                        sigma *= 0.9
                # Decrease base step size
                sigma *= 0.7
        
        return best_x, best_f
# EVOLVE-BLOCK-END
