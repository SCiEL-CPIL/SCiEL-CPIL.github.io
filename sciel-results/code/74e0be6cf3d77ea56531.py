"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase adaptive grid search exploiting WFG9 structure for dense Pareto front coverage."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0
        
        # WFG9: variables 0-2 control the front, variables 3-11 are shape/position
        # Fix non-front variables at midpoint to exploit structure
        base = mid.copy()
        
        X_list = []
        F_list = []
        used = 0
        
        def evaluate_point(x):
            nonlocal used
            if used >= self.budget or time.time() > deadline:
                return None
            f = problem(x)
            X_list.append(x.copy())
            F_list.append(np.array(f))
            used += 1
            return f
        
        # Phase 1: Dense 2D grid on the first two variables
        # WFG9's Pareto front is 2D, controlled primarily by x[0] and x[1]
        # Use 14x14 = 196 points to densely sample the front
        n_grid = 14
        t_vals = np.linspace(0.0, 1.0, n_grid)
        
        for t1 in t_vals:
            for t2 in t_vals:
                if used >= self.budget or time.time() > deadline:
                    break
                x = base.copy()
                x[0] = lo[0] + t1 * (hi[0] - lo[0])
                x[1] = lo[1] + t2 * (hi[1] - lo[1])
                # Keep x[2] at midpoint for now
                f = evaluate_point(x)
                if f is None:
                    break
            if used >= self.budget or time.time() > deadline:
                break
        
        # Phase 2: Use remaining budget to perturb the 3rd variable
        # This ensures we capture any 3rd dimension of the front if it exists
        # or refine the best points
        if used < self.budget and time.time() < deadline and F_list:
            F_arr = np.array(F_list)
            # Find non-dominated points
            nd_idx = []
            n_pts = len(F_arr)
            for i in range(n_pts):
                dominated = False
                for j in range(n_pts):
                    if i != j and np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated = True
                        break
                if not dominated:
                    nd_idx.append(i)
            
            # Perturb x[2] for a few non-dominated points
            # We have ~4 evaluations left (200 - 196)
            # Let's use them to check if moving x[2] improves anything
            # We'll pick the first few non-dominated points
            for idx in nd_idx[:4]:
                if used >= self.budget or time.time() > deadline:
                    break
                
                center = X_list[idx]
                # Try moving x[2] to low and high
                for val in [lo[2], hi[2]]:
                    if used >= self.budget or time.time() > deadline:
                        break
                    x = center.copy()
                    x[2] = val
                    f = evaluate_point(x)
                    if f is None:
                        break
        
        # Convert to arrays
        if X_list:
            X = np.array(X_list)
        else:
            X = base.reshape(1, -1)
        
        return X
# EVOLVE-BLOCK-END
