"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0
        
        # Strategy: Structured grid search over a 3D manifold
        # WFG9 is known to have specific parameter dependencies. 
        # We assume the first 3 dimensions are the most critical for shaping the Pareto front.
        # We fix dimensions 3..11 to their midpoints to reduce the effective search space from 12 to 3.
        
        # Budget allocation:
        # We have 200 evaluations.
        # Let's allocate 125 evaluations to a dense 5x5x5 grid.
        # This leaves 75 evaluations for a second, finer grid or random sampling if time permits.
        # However, a single dense grid is more robust for hypervolume if it covers the front well.
        
        # Let's try a 6x6x6 grid = 216 points. Too many.
        # 5x5x5 = 125 points.
        # Remaining 75 points: We can use them to refine the grid or do a second grid.
        # Alternatively, a 4x4x4 grid = 64 points, leaving 136 for more.
        
        # Let's stick to a high-quality dense grid on the first 3 dimensions.
        # 5x5x5 = 125 evaluations.
        
        n_grid = 5
        grid_1d = np.linspace(lo[0], hi[0], n_grid)
        
        X_list = []
        F_list = []
        
        # Generate grid points for dims 0, 1, 2
        # The rest are fixed to mid
        for i in range(n_grid):
            for j in range(n_grid):
                for k in range(n_grid):
                    if len(X_list) >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    x = np.full(self.dim, mid)
                    x[0] = grid_1d[i]
                    x[1] = grid_1d[j]
                    x[2] = grid_1d[k]
                    
                    # Ensure x is within bounds (linspace should be fine, but just in case)
                    # x = np.clip(x, lo, hi)
                    
                    X_list.append(x)
                    F_list.append(problem(x))
                    
                    if len(X_list) >= self.budget:
                        break
                if len(X_list) >= self.budget:
                    break
            if len(X_list) >= self.budget:
                break
        
        # If we have budget left, we can add some random points or refine the grid.
        # But for this strategy, we rely on the grid.
        # Let's use remaining budget for a finer grid in the center or random sampling to catch extremes.
        
        remaining_budget = self.budget - len(X_list)
        
        if remaining_budget > 0:
            # Use remaining budget for random sampling to ensure we don't miss anything
            # or to refine specific areas.
            # Random sampling with fixed seed is deterministic.
            for _ in range(remaining_budget):
                if time.time() > deadline:
                    break
                x = lo + (hi - lo) * np.random.random(self.dim)
                X_list.append(x)
                F_list.append(problem(x))
        
        # Return the last X (or all X, but the evaluator records all calls)
        # The return value is ignored, but we must return something.
        return X_list[-1] if X_list else None
# EVOLVE-BLOCK-END
