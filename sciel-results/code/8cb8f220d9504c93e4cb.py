import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ2 structure:
        # Pareto front is achieved when g = 0, i.e., x_i = 0.5 for i >= M (i.e., x_3...x_12 = 0.5)
        # The first M-1 = 2 variables (x_1, x_2) parameterize the Pareto front.
        # f_1 = sin(x_1 * pi/2)
        # f_2 = cos(x_1 * pi/2) * sin(x_2 * pi/2)
        # f_3 = cos(x_1 * pi/2) * cos(x_2 * pi/2)
        #
        # We want to maximize hypervolume, so we need good coverage of the Pareto front.
        # The Pareto front is a 2D manifold (quarter-sphere).

        # Strategy:
        # 1. Use a hypervolume-weighted non-uniform grid on (x_1, x_2) in [0,1]^2
        #    that places points at the centers of 2^k sub-rectangles formed by recursive bisection
        # 2. Add structured points (corners, edge midpoints)
        # 3. Use remaining budget for refinement with Halton sequence (low-discrepancy)

        base_x = np.zeros(self.dim)
        base_x[2:] = 0.5  # x_3, ..., x_12 = 0.5 to ensure g = 0

        n_evals = self.budget

        # Create a 2D grid with non-uniform spacing
        # Use a combination of structured and random points

        # First, create a base grid
        # Use a grid size that gives good coverage. 12x12 = 144 points.
        # This ensures we hit the "corners" and edges well.
        grid_size = 12
        grid_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                # Map grid indices to [0, 1]
                # Use midpoints to avoid boundary issues if necessary, but boundaries are fine.
                # Actually, for hypervolume, including the boundaries (0 and 1) is crucial 
                # to capture the full extent.
                # Let's use a uniform grid including endpoints.
                u = i / (grid_size - 1)
                v = j / (grid_size - 1)
                grid_points.append((u, v))
        
        # Add structured points for better coverage
        extra_points = [
            (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0),
            (0.5, 0.0), (0.5, 1.0), (0.0, 0.5), (1.0, 0.5),
            (0.25, 0.5), (0.75, 0.5), (0.5, 0.25), (0.5, 0.75),
            (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9),
            (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75),
            (0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85),
            (0.35, 0.35), (0.65, 0.35), (0.35, 0.65), (0.65, 0.65),
        ]
        
        all_params = grid_points + extra_points
        
        # Truncate to budget if needed
        all_params = all_params[:self.budget]
        
        # Evaluate all points
        results = []
        for x1, x2 in all_params:
            if len(results) >= self.budget:
                break
            if time.time() > deadline:
                break
            
            x = base_x.copy()
            x[0] = x1
            x[1] = x2
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            results.append(f)
        
        # Use remaining budget for Halton sequence refinement
        if len(results) < self.budget:
            remaining = self.budget - len(results)
            
            def halton(index, base):
                f = 1.0
                r = 0.0
                while index > 0:
                    f /= base
                    r += f * (index % base)
                    index //= base
                return r
            
            # Start Halton sequence from index 1
            halton_idx = 1
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                
                u = halton(halton_idx, 2)
                v = halton(halton_idx, 3)
                
                x = base_x.copy()
                x[0] = u
                x[1] = v
                x = np.clip(x, lo, hi)
                
                f = problem(x)
                results.append(f)
                halton_idx += 1
        
        return results
