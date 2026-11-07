"""Rewritten program for DTLZ7 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from numpy.polynomial.legendre import leggauss


class Optimizer:
    """
    Multi-objective optimizer for DTLZ7 using Gauss-Legendre quadrature
    to maximize hypervolume coverage of the Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        
        # Reserve 4 points for the corners of the Pareto front parameter space [0,1]^2
        # The corners are critical for hypervolume as they define the extremes.
        n_corners = 4
        budget_for_grid = self.budget - n_corners
        
        # Determine the size of the square grid N x N
        # We want N*N <= budget_for_grid
        N = int(np.sqrt(budget_for_grid))
        
        # If we have extra budget, we can potentially use a rectangular grid or add more points.
        # For now, we stick to a square grid and use remaining budget for edge midpoints if possible.
        
        # Generate Gauss-Legendre nodes in [0, 1]
        if N > 0:
            nodes, _ = leggauss(N)
            nodes = 0.5 * (nodes + 1.0)
        else:
            nodes = np.array([])
            
        # Create meshgrid of f1, f2 values
        f1_vals, f2_vals = np.meshgrid(nodes, nodes, indexing='ij')
        f1_flat = f1_vals.flatten()
        f2_flat = f2_vals.flatten()
        
        evaluated = 0
        
        # 1. Evaluate grid points
        for f1, f2 in zip(f1_flat, f2_flat):
            if evaluated >= self.budget:
                break
            x = np.zeros(n)
            x[0] = f1
            x[1] = f2
            x[2:] = 0.5
            x = np.clip(x, lo, hi)
            problem(x)
            evaluated += 1
            
        # 2. Evaluate corners
        corners = [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0)
        ]
        
        for f1, f2 in corners:
            if evaluated >= self.budget:
                break
            x = np.zeros(n)
            x[0] = f1
            x[1] = f2
            x[2:] = 0.5
            x = np.clip(x, lo, hi)
            problem(x)
            evaluated += 1
            
        # 3. If budget remains, add edge midpoints
        # This helps capture the shape of the front along the boundaries better.
        if evaluated < self.budget:
            midpoints = [
                (0.5, 0.0),
                (0.5, 1.0),
                (0.0, 0.5),
                (1.0, 0.5)
            ]
            for f1, f2 in midpoints:
                if evaluated >= self.budget:
                    break
                x = np.zeros(n)
                x[0] = f1
                x[1] = f2
                x[2:] = 0.5
                x = np.clip(x, lo, hi)
                problem(x)
                evaluated += 1
                
        # 4. If still budget remains, we could refine the grid or add more points.
        # For budget=200, N=14, 14*14=196, +4 corners = 200. So this part won't trigger.
        # But if budget was 204, N=14, 196+4=200, 4 left for midpoints.
        
        return None
# EVOLVE-BLOCK-END
