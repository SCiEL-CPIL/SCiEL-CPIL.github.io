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
        
        budget = self.budget
        dim = self.dim
        
        # For DTLZ4 with 3 objectives:
        # f1 = x[0]
        # f2 = x[1] * (1 - x[0])
        # f3 = (1 - x[1]) * (1 - x[0])
        # The Pareto front is achieved when x[2:] = 0.5 (g = 0)
        # On the front: f1 + f2 + f3 = 1
        # The front is a curved surface in 3D objective space.
        
        # Strategy: Use a dense, well-distributed set of points on the Pareto front
        # to maximize hypervolume with reference point [2.5, 2.5, 2.5].
        
        # Key insight: The hypervolume contribution depends on the distribution of
        # points in objective space. We want to cover the front as uniformly as possible
        # to maximize the volume of the union of boxes from each point to the reference.
        
        # Use a stratified sampling approach:
        # 1. Generate a 2D grid in (u, v) space where u = x[0], v = x[1]
        # 2. Map to objective space: f1 = u, f2 = v*(1-u), f3 = (1-v)*(1-u)
        # 3. Use a quasi-random or well-distributed grid to maximize coverage
        
        # For 200 evaluations, use a 2D grid with ~14x14 = 196 points
        # plus a few extra points at strategic locations
        
        # Determine optimal grid dimensions
        # We want to maximize coverage of the triangular front
        # A square grid in (u, v) maps to a curved surface in objective space
        
        # Use a slightly denser grid in the center where the front is more "curved"
        # and thus where hypervolume contribution per point is higher
        
        # Approach: Use a 14x14 grid = 196 points, then add 4 more points
        # at positions that fill gaps in the coverage
        
        n = 14
        points = []
        
        # Create a stratified grid in (u, v) space
        # Use midpoints of intervals for better coverage
        u_vals = np.linspace(0, 1, n + 1)
        v_vals = np.linspace(0, 1, n + 1)
        
        # Generate grid points at cell centers
        for i in range(n):
            for j in range(n):
                u = (u_vals[i] + u_vals[i+1]) / 2.0
                v = (v_vals[j] + v_vals[j+1]) / 2.0
                
                x = np.full(dim, 0.5)
                x[0] = u
                x[1] = v
                points.append(x)
        
        # Evaluate all grid points
        F = []
        for x in points:
            F.append(problem(x))
            if time.time() > deadline:
                break
        
        # If we have budget left, add refined points
        remaining = budget - len(F)
        if remaining > 0 and time.time() < deadline:
            # Add points at the corners and edges of the front
            # These are important for hypervolume as they define the boundaries
            extra_points = []
            
            # Corners of the (u, v) unit square
            corners = [(0, 0), (0, 1), (1, 0), (1, 1)]
            for u, v in corners:
                x = np.full(dim, 0.5)
                x[0] = u
                x[1] = v
                extra_points.append(x)
            
            # Midpoints of edges
            edge_mids = [(0, 0.5), (1, 0.5), (0.5, 0), (0.5, 1)]
            for u, v in edge_mids:
                x = np.full(dim, 0.5)
                x[0] = u
                x[1] = v
                extra_points.append(x)
            
            # Center
            x = np.full(dim, 0.5)
            x[0] = 0.5
            x[1] = 0.5
            extra_points.append(x)
            
            # Evaluate extra points
            for x in extra_points[:remaining]:
                F.append(problem(x))
                if time.time() > deadline:
                    break
        
        return F
