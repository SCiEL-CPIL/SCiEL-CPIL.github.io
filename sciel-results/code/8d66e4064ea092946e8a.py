"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-scale hierarchical optimizer for Schwefel-like functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit
        
        # Track best
        best_x = None
        best_f = np.inf
        call_count = 0
        
        def evaluate(x):
            nonlocal best_x, best_f, call_count
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Phase 1: Coarse grid search to find a good region
        # For Schwefel in 5D, the global optimum is typically around -4.20968 in each dimension
        # But due to hidden shifts/rotations, we need to search broadly first
        
        # Start with a coarse grid: 3 points per dimension = 3^5 = 243 evaluations
        # This gives us a good initial estimate of the landscape
        grid_points = 3
        grid_vals = np.linspace(-5.0, 5.0, grid_points)
        
        # Generate all grid points
        grid_coords = np.array(np.meshgrid(*([grid_vals] * dim)), dtype=np.float64)
        grid_coords = grid_coords.reshape(dim, -1).T
        
        for pt in grid_coords:
            if call_count >= budget or time.time() > deadline:
                break
            evaluate(pt)
        
        # Phase 2: Local refinement around the best point found
        # Use a fine grid around the best point
        if best_x is not None:
            # Determine refinement radius based on grid spacing
            grid_spacing = 10.0 / (grid_points - 1)  # 5.0
            refine_radius = grid_spacing
            
            # Fine grid: 5 points per dimension in the refinement region
            fine_grid_points = 5
            fine_grid_vals = np.linspace(-refine_radius, refine_radius, fine_grid_points)
            
            fine_coords = np.array(np.meshgrid(*([fine_grid_vals] * dim)), dtype=np.float64)
            fine_coords = fine_coords.reshape(dim, -1).T
            
            for offset in fine_coords:
                if call_count >= budget or time.time() > deadline:
                    break
                candidate = best_x + offset
                evaluate(candidate)
            
            # Phase 3: Even finer local search with random perturbations
            # Use (1+1)-ES style with adaptive step size
            if call_count < budget and time.time() < deadline:
                x = best_x.copy()
                fx = best_f
                sigma = refine_radius / fine_grid_points  # small step
                
                for _ in range(budget - call_count):
                    if time.time() > deadline:
                        break
                    
                    y = x + sigma * np.random.default_rng(42 + call_count).standard_normal(dim)
                    fy = evaluate(y)
                    
                    if fy < fx:
                        x, fx = y, fy
                        sigma *= 1.2
                    else:
                        sigma *= 0.85
                    
                    if sigma < 1e-8:
                        break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
