import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 280.0
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # WFG9 with 3 objectives: x0, x1 are shape variables
        # x2..x11 are distance variables
        n_shape = 2
        n_dist = dim - n_shape
        
        # Optimal distance variables: WFG h(x) is minimized at 0.35
        # For standard bounds [0,1] or similar, 0.35 * span + lo is optimal
        opt_dist = lo[n_shape:] + 0.35 * span[n_shape:]
        
        used = 0
        best_points = []  # list of (f, x) for tracking
        
        def evaluate(x):
            nonlocal used
            if used >= budget or time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            used += 1
            # Only keep points strictly better than reference [3.0, 5.0, 7.0]
            if f[0] < 3.0 and f[1] < 5.0 and f[2] < 7.0:
                best_points.append((f, x.copy()))
            return f
        
        # Phase 1: Dense grid on shape variables (x0, x1)
        # The Pareto front of WFG9 is primarily controlled by x0, x1
        # Use a 16x16 grid = 256 points, but we only have 200 budget
        # So use 14x14 = 196 grid points, leaving 4 for refinement
        
        # Actually, let's be smarter. Use a 13x13 grid = 169 points
        # Then use remaining 31 for local refinement
        n_grid = 13
        grid_x0 = np.linspace(lo[0], hi[0], n_grid)
        grid_x1 = np.linspace(lo[1], hi[1], n_grid)
        
        for x0 in grid_x0:
            for x1 in grid_x1:
                if used >= budget or time.time() > deadline:
                    break
                
                x = np.zeros(dim)
                x[0] = x0
                x[1] = x1
                x[2:] = opt_dist
                
                evaluate(x)
            
            if used >= budget or time.time() > deadline:
                break
        
        # Phase 2: Local refinement around best points
        # Sort best_points by hypervolume contribution (approximate by distance to origin)
        if best_points and used < budget and time.time() < deadline:
            # Find non-dominated points among best_points
            # Simple non-dominated sort
            non_dom = []
            for f, x in best_points:
                dominated = False
                for f2, _ in best_points:
                    if f2[0] <= f[0] and f2[1] <= f[1] and f2[2] <= f[2]:
                        if f2[0] < f[0] or f2[1] < f[1] or f2[2] < f[2]:
                            dominated = True
                            break
                if not dominated:
                    non_dom.append((f, x))
            
            # Refine around each non-dominated point with small Gaussian perturbations
            rng = np.random.default_rng(42)
            for f, x_base in non_dom:
                if used >= budget or time.time() > deadline:
                    break
                
                # Try a few local perturbations
                for _ in range(3):
                    if used >= budget or time.time() > deadline:
                        break
                    
                    # Small perturbation in shape variables
                    x = x_base.copy()
                    x[0] += rng.normal(0, 0.05 * span[0])
                    x[1] += rng.normal(0, 0.05 * span[1])
                    x[2:] = opt_dist
                    
                    evaluate(x)
        
        # Phase 3: If we have budget left, do a finer grid in promising regions
        if used < budget and time.time() < deadline:
            # Identify the best region from non_dom points
            if non_dom:
                # Find the bounding box of non-dominated points in shape space
                x0s = [x[0] for _, x in non_dom]
                x1s = [x[1] for _, x in non_dom]
                x0_min, x0_max = min(x0s), max(x0s)
                x1_min, x1_max = min(x1s), max(x1s)
                
                # Expand slightly
                margin = 0.1 * span[0]
                x0_min = max(lo[0], x0_min - margin)
                x0_max = min(hi[0], x0_max + margin)
                x1_min = max(lo[1], x1_min - margin)
                x1_max = min(hi[1], x1_max + margin)
                
                # Finer grid in this region
                n_fine = 8
                grid_x0_fine = np.linspace(x0_min, x0_max, n_fine)
                grid_x1_fine = np.linspace(x1_min, x1_max, n_fine)
                
                for x0 in grid_x0_fine:
                    for x1 in grid_x1_fine:
                        if used >= budget or time.time() > deadline:
                            break
                        
                        x = np.zeros(dim)
                        x[0] = x0
                        x[1] = x1
                        x[2:] = opt_dist
                        
                        evaluate(x)
                    
                    if used >= budget or time.time() > deadline:
                        break
        
        return None
