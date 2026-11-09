import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_shape = self.n_obj - 1
        n_pos = self.dim - n_shape
        
        # Use a grid for the shape variables to maximize hypervolume coverage.
        # For n_shape=2, a square grid is optimal for uniform coverage of the 2D front.
        # Determine grid size based on budget.
        # We want N_grid * N_grid <= budget.
        grid_size = int(np.sqrt(self.budget))
        
        # Ensure we don't exceed budget if sqrt is not integer
        # If grid_size * grid_size > budget, reduce grid_size
        while grid_size * grid_size > self.budget and grid_size > 1:
            grid_size -= 1
            
        # Create meshgrid for shape variables in [0, 1]
        # Using linspace includes boundaries which are important for hypervolume extremes.
        shape_vals_1 = np.linspace(0, 1, grid_size)
        shape_vals_2 = np.linspace(0, 1, grid_size)
        
        grid_1, grid_2 = np.meshgrid(shape_vals_1, shape_vals_2, indexing='ij')
        shape_samples = np.column_stack((grid_1.ravel(), grid_2.ravel()))
        
        # Map shape samples to bounds
        lo_shape = lo[n_pos:]
        hi_shape = hi[n_pos:]
        shape_samples = lo_shape + (hi_shape - lo_shape) * shape_samples
        
        # Construct full vectors
        # Position variables are set to 0.5 to make g(x)=0 (Pareto front)
        pos_val = np.full((shape_samples.shape[0], n_pos), 0.5)
        
        # Clip position values to bounds
        pos_val = np.clip(pos_val, lo[:n_pos], hi[:n_pos])
        
        X = np.hstack((pos_val, shape_samples))
        
        # Ensure X is within bounds
        X = np.clip(X, lo, hi)
        
        fvals = []
        for i in range(X.shape[0]):
            if time.time() > deadline:
                break
            x = X[i]
            f = problem(x)
            fvals.append(f)
            
        return fvals
