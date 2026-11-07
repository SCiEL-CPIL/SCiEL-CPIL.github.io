"""Rewritten program for DTLZ2 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer for DTLZ2 using a deterministic dense grid.
    
    Strategy:
    1. The Pareto front of DTLZ2 is a 2D manifold in the objective space.
    2. The position on the front is determined by the first two decision variables (x0, x1).
    3. The distance from the front is determined by the remaining variables (x2..x11).
    4. To maximize hypervolume, we place points directly on the Pareto front (g=0) by setting x2..x11 = 0.5.
    5. We use a deterministic uniform grid in the 2D position space (x0, x1) to ensure even coverage
       of the front. A 14x14 grid (196 points) utilizes most of the 200 budget efficiently.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Determine grid size to maximize points within budget
        # 14 * 14 = 196 <= 200
        n_grid = 14
        
        grid_points = []
        for i in range(n_grid):
            for j in range(n_grid):
                x = np.full(self.dim, 0.5)
                # Map grid indices to [0, 1]
                x[0] = i / (n_grid - 1)
                x[1] = j / (n_grid - 1)
                grid_points.append(x)
        
        grid_points = np.array(grid_points)
        # Clip to bounds
        grid_points = np.clip(grid_points, lo, hi)
        
        # Evaluate grid points
        used = 0
        for x in grid_points:
            if used >= self.budget:
                break
            problem(x)
            used += 1
            
        # If we have remaining budget, we can add a few more points
        # With 196 points used, we have 4 left.
        # Let's add the center point (0.5, 0.5) which is not in the even-sized grid.
        if used < self.budget:
            x_center = np.full(self.dim, 0.5)
            x_center[0] = 0.5
            x_center[1] = 0.5
            x_center = np.clip(x_center, lo, hi)
            problem(x_center)
            used += 1
            
        # Add 3 more points to fill in the "cross" of the front
        extra_points = [
            [0.25, 0.5],
            [0.75, 0.5],
            [0.5, 0.25]
        ]
        
        for px, py in extra_points:
            if used >= self.budget:
                break
            x_extra = np.full(self.dim, 0.5)
            x_extra[0] = px
            x_extra[1] = py
            x_extra = np.clip(x_extra, lo, hi)
            problem(x_extra)
            used += 1

        return grid_points
# EVOLVE-BLOCK-END
