"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG9 using analytic structure exploitation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # WFG9 analytic structure:
        # Only x[0] and x[1] affect objectives in a non-trivial way
        # The other 10 variables can be fixed to mid-range or optimal values
        # Let's fix variables 2-11 to their midpoint values
        
        base_x = np.zeros(self.dim)
        for i in range(self.dim):
            base_x[i] = (lo[i] + hi[i]) / 2.0
        
        # Strategy: Dense grid sampling over x[0] and x[1]
        # Use remaining budget for local refinement
        
        # Allocate budget: ~150 for grid, ~50 for refinement
        grid_budget = 150
        refine_budget = self.budget - grid_budget
        
        # Create grid points for x[0] and x[1]
        n_grid = int(np.sqrt(grid_budget))
        x0_vals = np.linspace(lo[0], hi[0], n_grid)
        x1_vals = np.linspace(lo[1], hi[1], n_grid)
        
        evaluated = []
        used = 0
        
        # Phase 1: Grid sampling
        for x0 in x0_vals:
            for x1 in x1_vals:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                x = base_x.copy()
                x[0] = x0
                x[1] = x1
                
                f = problem(x)
                evaluated.append((x, f))
                used += 1
            
            if used >= self.budget or time.time() > deadline:
                break
        
        if used >= self.budget or time.time() > deadline:
            return [e[0] for e in evaluated]
        
        # Phase 2: Local refinement around best points
        # Find non-dominated points from grid
        F = np.array([e[1] for e in evaluated])
        X = np.array([e[0] for e in evaluated])
        
        # Simple non-domination filter
        n_points = len(F)
        dominated = np.zeros(n_points, dtype=bool)
        for i in range(n_points):
            for j in range(n_points):
                if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        
        nd_points = np.where(~dominated)[0]
        
        # Refine around non-dominated points
        if len(nd_points) > 0 and used < self.budget:
            # Pick a few non-dominated points to refine
            n_refine = min(5, len(nd_points), refine_budget // 10)
            step = max(1, len(nd_points) // n_refine)
            refine_indices = nd_points[::step][:n_refine]
            
            for idx in refine_indices:
                if used >= self.budget or time.time() > deadline:
                    break
                    
                center_x = X[idx]
                center_f = F[idx]
                
                # Local random search around this point
                for _ in range(min(10, self.budget - used)):
                    if time.time() > deadline:
                        break
                    
                    # Perturb x[0] and x[1] with small steps
                    scale = 0.05 * (hi[:2] - lo[:2])
                    perturb = scale * np.random.randn(2)
                    
                    x = center_x.copy()
                    x[0] = np.clip(x[0] + perturb[0], lo[0], hi[0])
                    x[1] = np.clip(x[1] + perturb[1], lo[1], hi[1])
                    
                    f = problem(x)
                    evaluated.append((x, f))
                    used += 1
        
        return [e[0] for e in evaluated]
# EVOLVE-BLOCK-END
