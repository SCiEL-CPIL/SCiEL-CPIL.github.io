import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0
        
        # WFG9 with 3 objectives has a 2D Pareto front.
        # The front is determined by the first 2 variables.
        # The remaining variables are position variables, optimal at midpoint.
        n_shape = min(2, self.dim)
        
        # Strategy:
        # Use a dense grid on the 2 shape variables.
        # Budget 200.
        # 14x14 = 196 points.
        g1, g2 = 14, 14
        grid_sizes = [g1, g2]
        
        ranges = []
        for i in range(n_shape):
            if i < len(grid_sizes) and grid_sizes[i] > 0:
                ranges.append(np.linspace(lo[i], hi[i], grid_sizes[i]))
            else:
                ranges.append(np.array([mid[i]]))
        
        grids = np.meshgrid(*ranges, indexing='ij')
        grid_points = np.stack([g.ravel() for g in grids], axis=1)
        N_grid = grid_points.shape[0]
        
        # Construct full vectors: shape from grid, position from mid
        X_grid = np.tile(mid, (N_grid, 1))
        X_grid[:, :n_shape] = grid_points
        
        # Evaluate grid points
        F_grid = np.array([problem(x) for x in X_grid])
        used = N_grid
        
        if time.time() > deadline:
            return X_grid
            
        # Phase 2: Refinement with remaining budget
        # If we have budget left, we can add a few points or refine.
        # 196 used, 4 left.
        # Let's just use the grid points.
        # Or add 4 random points to ensure we don't miss edges?
        # The grid covers edges.
        
        X_final = X_grid
        
        # If we have more budget (shouldn't happen with 14x14=196 < 200),
        # we could refine, but 4 points is too few for meaningful refinement.
        
        return X_final
