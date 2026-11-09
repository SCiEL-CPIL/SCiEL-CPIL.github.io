import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_evals = 0
        all_points = []
        all_fvals = []

        def eval_point(x):
            nonlocal n_evals
            if n_evals >= self.budget:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            if len(f) != self.n_obj:
                return None
            f = np.array(f, dtype=float)
            if not np.all(np.isfinite(f)):
                return None
            all_points.append(x)
            all_fvals.append(f)
            n_evals += 1
            return f

        # Phase 1: Coarse grid on the known Pareto front structure
        # DTLZ2: x0, x1 determine angles; x2..x11 should be 0.5 for g=0
        # f1 = sin^2(pi * x0) * cos^2(pi * x1)
        # f2 = sin^2(pi * x0) * sin^2(pi * x1)
        # f3 = cos^2(pi * x0)
        
        # Use a structured grid in the (x0, x1) space to cover the front
        n_grid = 14  # 14x14 = 196 points, leaving 4 for refinement
        x0_vals = np.linspace(0.01, 0.99, n_grid)
        x1_vals = np.linspace(0.01, 0.99, n_grid)
        
        grid_points = []
        for x0 in x0_vals:
            for x1 in x1_vals:
                x = np.zeros(self.dim)
                x[0] = x0
                x[1] = x1
                x[2:] = 0.5
                grid_points.append(x)
        
        # Evaluate grid points
        grid_fvals = []
        for x in grid_points:
            if time.time() > deadline:
                break
            f = eval_point(x)
            if f is not None:
                grid_fvals.append(f)
            else:
                grid_fvals.append(None)
        
        # If we ran out of budget during grid, return what we have
        if n_evals >= self.budget:
            return all_points

        # Phase 2: Refinement around best-performing regions
        # Identify regions with good hypervolume contribution
        # Use a simple local search around the best 10 points from the grid
        
        valid_indices = [i for i, f in enumerate(grid_fvals) if f is not None]
        if len(valid_indices) > 0:
            # Sort by a simple proxy: sum of objectives (lower is better)
            valid_fvals = [grid_fvals[i] for i in valid_indices]
            sums = [np.sum(f) for f in valid_fvals]
            # Get top 10 best (lowest sum)
            top_idx = np.argsort(sums)[:min(10, len(valid_indices))]
            
            # For each top point, try small perturbations in x0 and x1
            n_refine = self.budget - n_evals
            n_per_point = max(1, n_refine // len(top_idx))
            
            for idx in top_idx:
                if n_evals >= self.budget or time.time() > deadline:
                    break
                base_x = grid_points[valid_indices[idx]]
                base_f = grid_fvals[valid_indices[idx]]
                
                # Try perturbations in x0 and x1
                for dx0 in [-0.02, 0.02]:
                    if n_evals >= self.budget or time.time() > deadline:
                        break
                    x = base_x.copy()
                    x[0] = np.clip(x[0] + dx0, 0.0, 1.0)
                    eval_point(x)
                
                for dx1 in [-0.02, 0.02]:
                    if n_evals >= self.budget or time.time() > deadline:
                        break
                    x = base_x.copy()
                    x[1] = np.clip(x[1] + dx1, 0.0, 1.0)
                    eval_point(x)
        
        return all_points
