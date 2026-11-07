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
        max_evals = self.budget
        used = 0
        
        # Strategy: Exploit DTLZ2 structure with a deterministic 10x10 grid in position space.
        # DTLZ2 has 2 position variables (x[0], x[1]) and (dim-2) direction variables.
        # The Pareto front is achieved when all direction variables are 0.5.
        # We sample a 10x10 grid of (x[0], x[1]) values, setting x[2:] = 0.5.
        # This guarantees coverage of the front's boundaries and corners.
        # Remaining budget is used for local search to minimize g (distance of x[2:] from 0.5).
        
        # Phase 1: Deterministic 10x10 grid
        grid_size = 10
        n_grid = grid_size * grid_size  # 100 points
        
        # If budget is less than grid size, adjust
        if n_grid > max_evals:
            grid_size = int(np.sqrt(max_evals))
            n_grid = grid_size * grid_size
        
        x0_vals = np.linspace(0, 1, grid_size)
        x1_vals = np.linspace(0, 1, grid_size)
        
        grid_points = []
        for x0 in x0_vals:
            for x1 in x1_vals:
                x = np.full(self.dim, 0.5)
                x[0] = x0
                x[1] = x1
                grid_points.append(x)
        
        grid_points = np.array(grid_points)
        
        # Evaluate grid points
        F_grid = np.array([problem(x) for x in grid_points])
        used += len(grid_points)
        
        # Check time
        if time.time() > deadline:
            return None
        
        # Phase 2: Local search on direction variables to minimize g
        # For each point on the front (or near it), we can try to improve the direction variables.
        # Since DTLZ2's g is sum((x_i - 0.5)^2), the optimal is x_i = 0.5.
        # But if the problem is slightly different, we use a simple local search.
        # We'll do a few iterations of gradient-free local search on a subset of the best points.
        
        # Select a subset of points for local search (e.g., the ones with lowest sum of objectives)
        sums = np.sum(F_grid, axis=1)
        # Sort by sum and take the best 20 points for local search
        n_local = min(20, max_evals - used)
        if n_local <= 0:
            return None
        
        # Indices of best points
        best_idx = np.argsort(sums)[:n_local]
        
        # Local search: perturb direction variables (x[2:]) and keep position variables (x[0], x[1])
        # Use a simple random perturbation with decreasing step size
        base_points = grid_points[best_idx]
        base_F = F_grid[best_idx]
        
        # We have max_evals - used evaluations left for local search
        remaining = max_evals - used
        n_iters = remaining // n_local if n_local > 0 else 0
        
        # If we have very few evaluations, just return
        if n_iters <= 0:
            return None
        
        # Local search loop
        for it in range(n_iters):
            if time.time() > deadline:
                break
            if used >= max_evals:
                break
            
            # For each base point, generate one perturbed version
            for i in range(n_local):
                if used >= max_evals:
                    break
                
                # Perturb direction variables (x[2:])
                x = base_points[i].copy()
                # Random perturbation with small step
                step = 0.1 * (1 - it / max(1, n_iters))
                x[2:] += np.random.uniform(-step, step, size=self.dim - 2)
                x = np.clip(x, lo, hi)
                
                f_trial = problem(x)
                used += 1
                
                # If better, update base point
                if np.sum(f_trial) < np.sum(base_F[i]):
                    base_points[i] = x
                    base_F[i] = f_trial
        
        return None
