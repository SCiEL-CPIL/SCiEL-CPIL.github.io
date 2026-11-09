"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Deterministic Grid Search with Adaptive Refinement for Sphere."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        # Initialize at center
        x = np.zeros(dim)
        fx = func(x)
        call_count = 1
        
        best_x = x.copy()
        best_fx = fx
        
        # Phase 1: Coarse grid search to find approximate basin
        # Use a small number of points per dimension to keep total calls manageable
        # For dim=5, 3^5 = 243 points. We have 500 budget, so this is feasible.
        # But we need to be careful. Let's use a smarter approach.
        
        # Since the minimum is in [-4, 4]^5, let's search there.
        # Coarse grid: 3 points per dim -> 3^5 = 243 calls.
        # If we use 5 points per dim -> 5^5 = 3125 calls. Too many.
        # Let's use 3 points per dim for the first phase.
        
        grid_points_per_dim = 3
        total_grid_points = grid_points_per_dim ** dim
        
        if call_count + total_grid_points <= budget:
            # Generate grid
            coords = np.linspace(-4.0, 4.0, grid_points_per_dim)
            grid = np.array(np.meshgrid(*([coords] * dim))).T.reshape(-1, dim)
            
            for point in grid:
                if call_count >= budget or time.time() > deadline:
                    break
                f_val = func(point)
                call_count += 1
                if f_val < best_fx:
                    best_fx = f_val
                    best_x = point.copy()
        
        # Phase 2: Local refinement around best_x
        # Use a local grid search with decreasing radius
        # This is like a deterministic version of Nelder-Mead or similar
        
        # Set up local refinement
        radius = 1.0  # Initial radius
        local_grid_points_per_dim = 3
        
        while call_count < budget and time.time() < deadline:
            # Generate local grid around best_x
            local_coords = np.linspace(-radius, radius, local_grid_points_per_dim)
            local_grid = np.array(np.meshgrid(*([local_coords] * dim))).T.reshape(-1, dim)
            local_grid += best_x
            
            # Clip to bounds
            local_grid = np.clip(local_grid, -5.0, 5.0)
            
            improved = False
            for point in local_grid:
                if call_count >= budget or time.time() > deadline:
                    break
                f_val = func(point)
                call_count += 1
                if f_val < best_fx:
                    best_fx = f_val
                    best_x = point.copy()
                    improved = True
            
            if not improved:
                # No improvement, shrink radius
                radius *= 0.5
                if radius < 1e-6:
                    break
            else:
                # Keep radius if improved, or shrink slightly
                radius *= 0.8
        
        # Phase 3: Final polish with very small steps
        if call_count < budget and time.time() < deadline:
            # Use gradient-free local search with very small steps
            step_size = 1e-4
            for _ in range(100):
                if call_count >= budget or time.time() > deadline:
                    break
                
                improved = False
                for i in range(dim):
                    for direction in [-1, 1]:
                        if call_count >= budget or time.time() > deadline:
                            break
                        
                        x_new = best_x.copy()
                        x_new[i] += direction * step_size
                        x_new = np.clip(x_new, -5.0, 5.0)
                        
                        f_new = func(x_new)
                        call_count += 1
                        
                        if f_new < best_fx:
                            best_fx = f_new
                            best_x = x_new.copy()
                            improved = True
                
                if not improved:
                    step_size *= 0.5
                    if step_size < 1e-8:
                        break
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
