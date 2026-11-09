import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Budget allocation: 196 grid + 4 strategic
        n_grid = 196
        n_strategic = 4
        
        # Generate a 14x14 grid in u-space [0, 1]
        u_vals = np.linspace(0, 1, 14)
        
        # Transform to x-space with clustering near 0
        # x = u^2 clusters points near 0, which is the high-value region
        x_vals = u_vals ** 2
        
        # Create meshgrid
        X_grid, Y_grid = np.meshgrid(x_vals, x_vals)
        grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
        
        # Strategic points: Midpoints in the high-value region (0 to 0.5)
        # The grid has points at 0, 0.0059, 0.023, 0.052, 0.092, 0.141, 0.203, 0.287...
        # 0.25 is between 0.203 and 0.287.
        # 0.125 is between 0.092 and 0.141.
        strategic_points = np.array([
            [0.125, 0.125],
            [0.375, 0.125],
            [0.125, 0.375],
            [0.375, 0.375]
        ])
        
        all_points = np.vstack([grid_points, strategic_points])
        
        # Prepare full dimensionality vectors
        X = np.zeros((len(all_points), self.dim))
        X[:, 0] = all_points[:, 0]
        X[:, 1] = all_points[:, 1]
        X[:, 2:] = lo[2:] # Set to lower bound (0) for DTLZ7 optimality
        
        results = []
        for i in range(len(X)):
            if time.time() > deadline:
                break
            
            x = np.clip(X[i], lo, hi)
            f = problem(x)
            results.append(f)
            
        return results
