"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Grid-based coarse-to-fine deterministic scan for Rastrigin-like functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        def grid_scan(center, size, func, dim):
            """
            Evaluate the centers of 2^dim sub-cells of a hypercube.
            Returns the best point and its value.
            """
            best_x = None
            best_f = float('inf')
            
            offset = size / 4.0
            for mask in range(2 ** dim):
                x = np.array([center[i] + (offset if (mask >> i) & 1 else -offset) for i in range(dim)])
                x = np.clip(x, -5.0, 5.0)
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            return best_x, best_f, size / 2.0

        # Stage 1: Coarse grid over the whole box
        # 2^5 = 32 points, cell size 5.0
        center = np.zeros(dim)
        size = 10.0
        best_x, best_f, size = grid_scan(center, size, func, dim)
        center = best_x
        calls = 2 ** dim

        # Stage 2-5: Refine the best region
        # Each stage: 32 points, cell size halves
        for _ in range(4):
            if calls + 2 ** dim > budget:
                break
            best_x, best_f, size = grid_scan(center, size, func, dim)
            center = best_x
            calls += 2 ** dim

        # Now size is 10 * (0.5)^5 = 0.3125
        # We have used 32 * 5 = 160 calls
        # Remaining: 500 - 160 = 340 calls
        
        remaining = budget - calls
        if remaining > 0:
            # Use Nelder-Mead for local refinement
            from scipy.optimize import minimize
            
            # Initial simplex: best_x plus small perturbations
            x0 = best_x.copy()
            simplex = [x0]
            # Add points with small offsets in each dimension
            for i in range(dim):
                dx = x0.copy()
                dx[i] += 0.1
                simplex.append(dx)
            
            # Use Nelder-Mead
            try:
                result = minimize(func, x0, method='Nelder-Mead', 
                                 options={'maxiter': remaining, 'xatol': 1e-8, 'fatol': 1e-10})
                if result.fun < best_f:
                    best_x, best_f = result.x, result.fun
            except:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
