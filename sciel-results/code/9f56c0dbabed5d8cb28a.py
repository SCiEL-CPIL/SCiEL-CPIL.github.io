import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        # Time guard: keep a safety margin before the 300s wall-clock limit.
        start_time = time.time()
        deadline = start_time + 270.0

        best_x = np.zeros(n)
        best_f = float('inf')
        evals_left = budget

        # Grid-based adaptive search
        # Start with a coarse grid over [-5, 5]^n
        # Each subdivision evaluates 2^n points (one per sub-cell center)
        
        # Define the current search region
        low = np.full(n, -5.0)
        high = np.full(n, 5.0)
        
        # Evaluate the initial coarse grid: 2^n points at cell centers
        # For n=5, that's 32 points
        num_subdivisions = 2
        points_per_level = num_subdivisions ** n
        
        # Generate grid points for a region
        def generate_grid_points(low, high, num_subdivisions):
            """Generate grid points at centers of sub-cells."""
            # Create indices for each dimension
            indices = np.arange(num_subdivisions)
            # Create all combinations
            grids = np.meshgrid(*([indices] * n), indexing='ij')
            grid_indices = np.stack(grids, axis=-1)
            grid_indices = grid_indices.reshape(-1, n)
            
            # Convert indices to coordinates (cell centers)
            cell_width = (high - low) / num_subdivisions
            centers = low + (grid_indices + 0.5) * cell_width
            return centers
        
        # First level: evaluate the coarse grid
        grid_points = generate_grid_points(low, high, num_subdivisions)
        
        if evals_left >= len(grid_points):
            for i in range(len(grid_points)):
                f = func(grid_points[i])
                if f < best_f:
                    best_f = f
                    best_x = grid_points[i].copy()
                evals_left -= 1
        else:
            # Not enough budget for full grid, evaluate what we can
            for i in range(evals_left):
                f = func(grid_points[i])
                if f < best_f:
                    best_f = f
                    best_x = grid_points[i].copy()
                evals_left -= 1
            return best_x, best_f

        # Now recursively subdivide the best cell
        # Find which cell contains the best point
        # The best point is at grid_points[argmin_f], so we know its cell index
        
        # Track the best cell for subdivision
        # We need to know which cell the best point came from
        
        # Let's restructure: keep track of cells and their best values
        # Level 1: evaluate all 2^n cells, find best cell
        # Level 2: subdivide best cell into 2^n sub-cells, evaluate them
        # Level 3: subdivide best sub-cell, etc.
        
        # Reset and redo with proper tracking
        # Actually, let's just do it iteratively with proper cell tracking
        
        # Start over with a cleaner approach
        # We already used some evals, so let's continue from where we are
        
        # The best point so far is best_x. We need to find which cell it's in.
        # But we didn't track that. Let's use a different approach.
        
        # Alternative: just keep subdividing around the best point found
        # Use a local grid refinement strategy
        
        # Current best is best_x. Create a local grid around it.
        # The initial grid had cell width = 10/2 = 5
        # Let's subdivide the cell containing best_x
        
        # Find the cell containing best_x in the original grid
        cell_width = (high - low) / num_subdivisions
        cell_indices = np.floor((best_x - low) / cell_width).astype(int)
        cell_indices = np.clip(cell_indices, 0, num_subdivisions - 1)
        
        # The best cell is defined by:
        best_low = low + cell_indices * cell_width
        best_high = best_low + cell_width
        
        # Now subdivide this cell into 2^n sub-cells
        while evals_left > 0 and time.time() < deadline:
            # Generate sub-grid for the current best cell
            sub_points = generate_grid_points(best_low, best_high, num_subdivisions)
            
            if evals_left < len(sub_points):
                # Not enough for full sub-grid, evaluate what we can
                for i in range(evals_left):
                    f = func(sub_points[i])
                    if f < best_f:
                        best_f = f
                        best_x = sub_points[i].copy()
                    evals_left -= 1
                break
            
            # Evaluate all sub-points
            local_best_f = float('inf')
            local_best_x = sub_points[0].copy()
            
            for i in range(len(sub_points)):
                f = func(sub_points[i])
                if f < best_f:
                    best_f = f
                    best_x = sub_points[i].copy()
                if f < local_best_f:
                    local_best_f = f
                    local_best_x = sub_points[i].copy()
                evals_left -= 1
            
            # Find which sub-cell contains the local best
            sub_cell_width = (best_high - best_low) / num_subdivisions
            sub_cell_indices = np.floor((local_best_x - best_low) / sub_cell_width).astype(int)
            sub_cell_indices = np.clip(sub_cell_indices, 0, num_subdivisions - 1)
            
            # Update the best cell to be the sub-cell containing the local best
            best_low = best_low + sub_cell_indices * sub_cell_width
            best_high = best_low + sub_cell_width
            
            # Check if we've converged (cell is very small)
            if np.all(best_high - best_low < 1e-6):
                break
        
        return best_x, best_f
