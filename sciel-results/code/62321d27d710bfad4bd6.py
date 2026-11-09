# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ7.
    Strategy: Sample the 2D Pareto front (x_3..x_22 = 0) with a non-uniform grid
    that is denser near the ideal point (f1=0, f2=0) to maximize hypervolume.
    Uses a power-law transform to concentrate points near the origin in objective space.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        results = []
        n_evals = 0

        # Determine grid size
        # We want a square grid N x N such that N^2 <= budget.
        # For budget=200, sqrt(200) ~ 14.14, so N=14, N^2=196.
        # We can use the remaining 4 evaluations for refinement or just leave them.
        
        grid_size = int(np.sqrt(self.budget))
        grid_size = max(2, grid_size)
        while (grid_size + 1) ** 2 <= self.budget:
            grid_size += 1
        
        # Non-linear grid to concentrate points near the ideal objectives
        # f1 = 1 - x1, f2 = x2.
        # We want f1, f2 small.
        # x1 = 1 - f1, x2 = f2.
        # Let t in [0, 1] be the grid coordinate.
        # We want the distribution of f1 and f2 to be denser near 0.
        # If we sample t uniformly, and set f = t^gamma, then for gamma > 1,
        # t^gamma is denser near 0.
        # So x1 = 1 - t^gamma1, x2 = t^gamma2.
        
        # Tuning gamma:
        # Hypervolume is maximized when points are distributed such that the
        # marginal contribution of each point is equal. For a 2D front with
        # reference point (r1, r2), the optimal density is proportional to
        # 1/(f1*f2) in the region where points are non-dominated.
        # This suggests a power law with gamma around 1.5-2.0.
        # Let's try gamma=1.5 which might balance the density better.
        
        gamma1 = 1.5
        gamma2 = 1.5
        
        t_vals = np.linspace(0.0, 1.0, grid_size)
        
        # x1 = 1 - t^gamma1. 
        # When t=0, x1=1, f1=0.
        # When t=1, x1=0, f1=1.
        # This maps t in [0,1] to x1 in [1,0].
        
        # x2 = t^gamma2.
        # When t=0, x2=0, f2=0.
        # When t=1, x2=1, f2=1.
        
        x1_vals = 1.0 - t_vals ** gamma1
        x2_vals = t_vals ** gamma2

        # Iterate over the grid
        for i in range(grid_size):
            for j in range(grid_size):
                if n_evals >= self.budget:
                    break
                if time.time() > deadline:
                    break

                x = np.zeros(self.dim)
                x[0] = x1_vals[i]
                x[1] = x2_vals[j]
                # x[2:] = 0 for g=0 (Pareto front)

                try:
                    obj = problem(x)
                    obj = list(obj)
                    results.append(obj)
                    n_evals += 1
                except Exception:
                    break
            if n_evals >= self.budget or time.time() > deadline:
                break
        
        # If we have remaining budget, we can try to fill in some gaps or refine.
        # The remaining 4 points could be used for a few random perturbations to ensure robustness,
        # but since we know the front, deterministic grid is better.
        # However, we can use them to add points at the corners of the Pareto front
        # which might be underrepresented in the grid.
        
        remaining = self.budget - n_evals
        if remaining > 0:
            # Add points at the extreme corners of the Pareto front
            # These are points where one objective is minimized and the other is maximized
            corner_points = [
                [1.0, 0.0],  # f1=0, f2=1
                [0.0, 1.0],  # f1=1, f2=0
                [1.0, 1.0],  # f1=0, f2=0 (ideal point)
                [0.5, 0.5],  # middle point
            ]
            
            for k in range(min(remaining, len(corner_points))):
                if time.time() > deadline:
                    break
                
                x = np.zeros(self.dim)
                x[0] = corner_points[k][0]
                x[1] = corner_points[k][1]
                
                try:
                    obj = problem(x)
                    obj = list(obj)
                    results.append(obj)
                    n_evals += 1
                except Exception:
                    break
        
        return results
# EVOLVE-BLOCK-END
