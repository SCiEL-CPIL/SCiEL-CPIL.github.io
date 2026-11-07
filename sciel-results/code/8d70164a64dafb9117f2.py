"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coarse-to-fine grid scan with adaptive refinement and local polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s kill

        def time_left():
            return deadline - time.time()

        dim = self.dim
        budget = self.budget
        calls_made = 0

        # We will track the best point found
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            calls_made += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Coarse grid scan
        # We want to use about 60% of the budget for the grid scan.
        # The grid size should be such that the number of points is manageable.
        # Let's try a grid of size 3^dim or 4^dim?
        # For dim=5, 3^5 = 243, 4^5 = 1024.
        # 243 is a good start. It covers the space with a step of 10/2 = 5.
        # Wait, the box is [-5, 5]^5.
        # If we use 3 points per dimension: -5, 0, 5.
        # If we use 5 points per dimension: -5, -2.5, 0, 2.5, 5.
        
        # Let's start with a coarse grid of 3 points per dimension.
        # This gives 243 evaluations.
        # Then we refine the best region.

        # Actually, let's be smarter.
        # We have 500 calls.
        # Let's use 125 calls for a 5^3 grid? No, 5^5 = 3125 is too many.
        # 3^5 = 243.
        # Let's use 3 points per dimension initially.
        
        grid_size = 3
        step = 10.0 / (grid_size - 1)
        coords = np.linspace(-5, 5, grid_size)
        
        # Generate all grid points
        grid_points = np.array(np.meshgrid(*([coords] * dim))).T.reshape(-1, dim)
        
        # Evaluate all grid points
        for p in grid_points:
            if calls_made >= budget or time_left() < 1.0:
                break
            evaluate(p)
        
        if best_x is None:
            # Fallback: random search
            rng = np.random.default_rng(0)
            for _ in range(budget):
                if time_left() < 1.0:
                    break
                x = rng.uniform(-5, 5, dim)
                evaluate(x)
            return best_x, best_f

        # Phase 2: Refine around the best point
        # We have used ~243 calls. Remaining: ~257 calls.
        # Let's do a local search around best_x.
        # We can use a smaller grid or a local optimization method.
        # Since it's a black-box, let's use a local grid refinement.
        
        # Determine the step size for refinement.
        # The previous step was 5.0. Let's try 1.0, then 0.2, etc.
        # Or use a local (1+1)-ES starting from best_x.
        
        # Let's use a hybrid: a few local grid refinements, then (1+1)-ES.
        
        # Local grid refinement 1: step = 1.0
        # Center at best_x.
        # Range: [best_x - 1, best_x + 1] clipped to [-5, 5].
        
        def local_grid_scan(center, step, n_points=3):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or time_left() < 1.0:
                return
            # Generate grid points around center
            offsets = np.linspace(-step, step, n_points)
            local_coords = [center[i] + offsets for i in range(dim)]
            local_points = np.array(np.meshgrid(*local_coords)).T.reshape(-1, dim)
            # Clip to box
            local_points = np.clip(local_points, -5, 5)
            
            for p in local_points:
                if calls_made >= budget or time_left() < 1.0:
                    break
                evaluate(p)
        
        # Refinement 1: step = 1.0, 3 points per dim -> 243 points (too many if we want to save for ES)
        # Let's use 2 points per dim (just the neighbors) or 3 points.
        # 3^5 = 243. We have ~257 left. So we can do one full refinement.
        # But maybe we should use fewer points to leave budget for ES.
        # Let's use 3 points per dim for the first refinement.
        
        local_grid_scan(best_x, 1.0, 3)
        
        # Refinement 2: step = 0.2, 3 points per dim -> 243 points.
        # If we have budget left.
        if calls_made < budget and time_left() > 1.0:
            local_grid_scan(best_x, 0.2, 3)
        
        # Phase 3: Local (1+1)-ES polish
        # Use remaining budget for a (1+1)-ES starting from best_x.
        rng = np.random.default_rng(1)
        x = best_x.copy()
        fx = best_f
        sigma = 0.1  # Start with a small step
        
        while calls_made < budget and time_left() > 0.5:
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = evaluate(y)
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
            else:
                sigma *= 0.9
                if sigma < 1e-6:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
