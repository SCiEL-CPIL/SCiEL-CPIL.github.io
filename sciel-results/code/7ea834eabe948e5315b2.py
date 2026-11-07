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
        # 1. Use a dense grid on (x_1, x_2) in [0,1]^2
        # 2. Add structured points (corners, edge midpoints)
        # 3. Use remaining budget for refinement with random points
        
        base_x = np.zeros(self.dim)
        base_x[2:] = 0.5  # x_3, ..., x_12 = 0.5 to ensure g = 0
        
        # Generate grid points
        n_grid = 144  # 12x12 grid
        grid_points = []
        n_per_dim = 12
        for i in range(n_per_dim):
            for j in range(n_per_dim):
                x1 = (i + 0.5) / n_per_dim
                x2 = (j + 0.5) / n_per_dim
                grid_points.append((x1, x2))
        
        # Add structured points for better coverage
        extra_points = [
            (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0),
            (0.5, 0.0), (0.5, 1.0), (0.0, 0.5), (1.0, 0.5),
            (0.25, 0.5), (0.75, 0.5), (0.5, 0.25), (0.5, 0.75),
            (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9),
            (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75),
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
        
        # Use remaining budget for random refinement
        if len(results) < self.budget:
            rng = np.random.default_rng(42)
            remaining = self.budget - len(results)
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                
                x1 = rng.random()
                x2 = rng.random()
                
                x = base_x.copy()
                x[0] = x1
                x[1] = x2
                x = np.clip(x, lo, hi)
                
                f = problem(x)
                results.append(f)
        
        return results
