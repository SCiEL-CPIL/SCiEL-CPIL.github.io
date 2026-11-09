"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for WFG9.
    Uses a deterministic grid over the first 3 variables (which directly determine
    the Pareto front) with remaining variables at midpoint, followed by a local
    search phase to refine the front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # 30s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0

        # WFG9: the first 3 variables determine the Pareto front.
        # We'll use a grid over the first 3 variables and fix the rest at midpoint.
        # Budget is 200, so we need to be careful with the grid size.
        
        # Let's use a grid of size g for each of the first 3 variables.
        # Total evaluations for grid: g^3
        # We want g^3 <= budget // 2 to leave room for refinement.
        # budget // 2 = 100, so g^3 <= 100 => g <= 4 (4^3=64), g=5 (125>100)
        # Let's try g=4 for the grid (64 points), leaving 136 for refinement.
        
        g = 4
        grid_points = []
        for i in range(g):
            for j in range(g):
                for k in range(g):
                    x = np.full(self.dim, 0.0)
                    x[0] = lo[0] + (hi[0] - lo[0]) * i / (g - 1)
                    x[1] = lo[1] + (hi[1] - lo[1]) * j / (g - 1)
                    x[2] = lo[2] + (hi[2] - lo[2]) * k / (g - 1)
                    for d in range(3, self.dim):
                        x[d] = mid[d]
                    grid_points.append(x)
        
        grid_points = np.array(grid_points)
        n_grid = len(grid_points)
        
        # Evaluate grid points
        F = np.zeros((n_grid, self.n_obj))
        for i in range(n_grid):
            if time.time() > deadline:
                break
            F[i] = problem(grid_points[i])
        
        used = n_grid
        
        # Local search phase: refine each grid point by perturbing the first 3 variables
        # and the rest slightly.
        # We'll do a simple coordinate descent on the first 3 variables for each grid point.
        # For each grid point, try perturbing each of the first 3 variables in +/- directions.
        # That's 3 * 2 = 6 evaluations per grid point per iteration.
        # With 136 budget left, we can do about 22 iterations of 6 evaluations.
        # But let's be more efficient: only refine the non-dominated points.
        
        # First, identify non-dominated points from the grid.
        def is_dominated(f, F_set):
            """Check if f is dominated by any point in F_set."""
            for f2 in F_set:
                if np.all(f2 <= f) and np.any(f2 < f):
                    return True
            return False
        
        # Keep track of all evaluated points
        X_all = [grid_points[i] for i in range(n_grid)]
        F_all = [F[i] for i in range(n_grid)]
        
        # Local search: for each non-dominated grid point, try to improve it
        # by perturbing the first 3 variables.
        step_sizes = [0.1, 0.05, 0.02, 0.01]
        
        for step in step_sizes:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            
            # Get non-dominated points
            nondom_idx = []
            for i in range(len(F_all)):
                dominated = False
                for j in range(len(F_all)):
                    if i != j and np.all(F_all[j] <= F_all[i]) and np.any(F_all[j] < F_all[i]):
                        dominated = True
                        break
                if not dominated:
                    nondom_idx.append(i)
            
            for idx in nondom_idx:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                x = X_all[idx].copy()
                f = F_all[idx].copy()
                
                # Try perturbing each of the first 3 variables
                for d in range(3):
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    for direction in [-1, 1]:
                        if used >= self.budget:
                            break
                        if time.time() > deadline:
                            break
                        
                        x_new = x.copy()
                        x_new[d] += direction * step * (hi[d] - lo[d])
                        x_new[d] = np.clip(x_new[d], lo[d], hi[d])
                        
                        f_new = problem(x_new)
                        used += 1
                        
                        # If the new point is better (non-dominated by the old point),
                        # replace the old point with the new one.
                        if not (np.all(f_new >= f) and np.any(f_new > f)):
                            # New point is not dominated by old point
                            # Check if new point dominates old point
                            if np.all(f_new <= f) and np.any(f_new < f):
                                X_all[idx] = x_new
                                F_all[idx] = f_new
                                x = x_new
                                f = f_new
                            else:
                                # Neither dominates the other; add the new point
                                X_all.append(x_new)
                                F_all.append(f_new)
        
        # Final non-dominated set
        final_F = np.array(F_all)
        final_X = np.array(X_all)
        
        return final_X
# EVOLVE-BLOCK-END
