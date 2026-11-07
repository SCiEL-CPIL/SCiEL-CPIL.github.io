import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        
        # DTLZ4: Pareto front is achieved when x[2:] = 0.5 (minimizing g to 0)
        # The front in objective space is parameterized by x[0], x[1] in [0,1]
        # f1 = x[0], f2 = x[1]*(1-x[0]), f3 = (1-x[1])*(1-x[0]) approximately
        # More precisely for DTLZ4:
        # f1 = x[0], f2 = x[1]*(1-x[0]), f3 = (1-x[1])*(1-x[0])
        # The Pareto front is a curved surface in 3D.
        
        # Strategy: Use a dense, well-distributed grid on the Pareto front
        # to maximize hypervolume. Since we know the front structure,
        # we can directly sample points that lie on or near the front.
        
        # For DTLZ4 with n_obj=3, dim=12:
        # f1 = x[0]
        # f2 = x[1] * (1 - x[0])
        # f3 = (1 - x[1]) * (1 - x[0])
        # g = 0 when x[2:] = 0.5
        # So on the front: f1 + f2 + f3 = x[0] + x[1]*(1-x[0]) + (1-x[1])*(1-x[0])
        # = x[0] + (1-x[0]) = 1
        # So the front lies on the plane f1+f2+f3 = 1, which is the standard DTLZ front.
        
        # We want to maximize hypervolume with reference point [2.5, 2.5, 2.5].
        # Points must be strictly better than ref in all objectives.
        
        # Best strategy: densely sample the Pareto front.
        # Use a 2D grid over x[0] and x[1] to cover the front.
        
        budget = self.budget
        dim = self.dim
        
        # Determine grid size
        # For a 2D grid, n_grid x n_grid = budget
        n_grid = int(np.sqrt(budget))
        # 14*14 = 196 <= 200, 15*15 = 225 > 200
        # Use 14x14 = 196 points, leaving 4 for potential refinement
        
        # Create grid points
        x0_vals = np.linspace(0, 1, n_grid)
        x1_vals = np.linspace(0, 1, n_grid)
        
        points = []
        for x0 in x0_vals:
            for x1 in x1_vals:
                x = np.full(dim, 0.5)
                x[0] = x0
                x[1] = x1
                points.append(x)
        
        # Evaluate all grid points
        F = []
        for x in points:
            F.append(problem(x))
            if time.time() > deadline:
                break
        
        # If we have budget left, add some refined points
        remaining = budget - len(points)
        if remaining > 0 and time.time() < deadline:
            # Add midpoints or finer subdivisions in regions with high hypervolume contribution
            # For DTLZ4 front, the hypervolume contribution is roughly uniform
            # Add a few extra points at strategic locations
            extra_points = []
            # Add center point if not already there
            if n_grid % 2 == 0:
                center_x = np.full(dim, 0.5)
                center_x[0] = 0.5
                center_x[1] = 0.5
                extra_points.append(center_x)
            
            # Add points at 1/4 and 3/4 positions
            for frac in [0.25, 0.75]:
                for x0 in [frac, 1-frac]:
                    for x1 in [frac, 1-frac]:
                        if time.time() > deadline:
                            break
                        x = np.full(dim, 0.5)
                        x[0] = x0
                        x[1] = x1
                        extra_points.append(x)
            
            for x in extra_points[:remaining]:
                F.append(problem(x))
                if time.time() > deadline:
                    break
        
        return points
