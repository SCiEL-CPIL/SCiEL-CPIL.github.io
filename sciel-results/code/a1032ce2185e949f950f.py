"""Optimizer for BBOB-style Rastrigin f3 with grid-based coarse-to-fine scan."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Grid-based coarse-to-fine deterministic scan for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0
        budget = self.budget
        dim = self.dim

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Coarse grid - 3 levels per dimension (3^5 = 243 points)
        # This covers [-5, 5]^5 with spacing ~2.5
        coarse_levels = np.array([-4.0, 0.0, 4.0])
        
        # Generate all combinations
        grid_points = list(np.meshgrid(*[coarse_levels] * dim, indexing='ij'))
        grid_points = np.array(list(zip(*[g.ravel() for g in grid_points])))
        
        # Evaluate all coarse grid points
        for p in grid_points:
            if calls_made >= budget or time.time() > deadline:
                break
            evaluate(p)
        
        if calls_made >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Fine grid around best region
        # Take the cell containing best_x and subdivide it
        # The coarse grid has cells of size ~2.5, so subdivide into 3x3x3x3x3 = 243
        # But that's too many. Instead, do a local fine grid of 3^5 around best_x
        
        # Determine the cell bounds for best_x
        # Coarse levels are at -4, 0, 4, so cells are [-5,-2], [-2,2], [2,5]
        # Find which cell best_x is in
        cell_centers = []
        for d in range(dim):
            if best_x[d] < -2.0:
                lo, hi = -5.0, -2.0
            elif best_x[d] < 2.0:
                lo, hi = -2.0, 2.0
            else:
                lo, hi = 2.0, 5.0
            cell_centers.append((lo, hi))
        
        # Create fine grid with 3 points per dimension in each cell
        fine_points = []
        for d in range(dim):
            lo, hi = cell_centers[d]
            fine_points.append(np.linspace(lo, hi, 3))
        
        # Generate all combinations
        fine_grid = list(np.meshgrid(*fine_points, indexing='ij'))
        fine_grid = np.array(list(zip(*[g.ravel() for g in fine_grid])))
        
        # Skip points already evaluated (within tolerance)
        for p in fine_grid:
            if calls_made >= budget or time.time() > deadline:
                break
            # Check if this point is close to an already-evaluated point
            # For simplicity, just evaluate all (some may be duplicates but that's OK)
            evaluate(p)
        
        if calls_made >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 3: Ultra-fine grid around the new best
        # Subdivide the cell containing the new best_x into 3^5 points
        cell_centers = []
        for d in range(dim):
            # Use a smaller neighborhood around best_x
            radius = 0.5
            lo = max(-5.0, best_x[d] - radius)
            hi = min(5.0, best_x[d] + radius)
            cell_centers.append((lo, hi))
        
        ultra_points = []
        for d in range(dim):
            lo, hi = cell_centers[d]
            ultra_points.append(np.linspace(lo, hi, 3))
        
        ultra_grid = list(np.meshgrid(*ultra_points, indexing='ij'))
        ultra_grid = np.array(list(zip(*[g.ravel() for g in ultra_grid])))
        
        for p in ultra_grid:
            if calls_made >= budget or time.time() > deadline:
                break
            evaluate(p)
        
        if calls_made >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 4: Local (1+1)-ES refinement from best point
        x = best_x.copy()
        fx = best_f
        sigma = 0.1
        no_improve = 0
        max_no_improve = 15
        
        while calls_made < budget and time.time() < deadline:
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x, fx = y, fy
                no_improve = 0
                sigma = min(sigma * 1.2, 0.3)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                no_improve += 1
                sigma = max(sigma * 0.85, 0.005)
            
            if no_improve >= max_no_improve:
                # Try a few kicks
                for _ in range(3):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    y = np.clip(x + 0.5 * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < fx:
                        x, fx = y, fy
                        sigma = 0.1
                        no_improve = 0
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        break
                else:
                    # No improvement, shrink more and reset counter
                    sigma = max(sigma * 0.7, 0.001)
                    no_improve = 0
        
        return best_x, best_f
# EVOLVE-BLOCK-END
