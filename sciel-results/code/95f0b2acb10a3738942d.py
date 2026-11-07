import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Phase 1: Random Initialization
        # Use 40% of budget for exploration
        init_count = min(40, self.budget // 2)
        if init_count < 10:
            init_count = 10
            
        X_init = np.random.uniform(lo, hi, size=(init_count, self.dim))
        F_init = np.array([problem(x) for x in X_init])
        
        all_X = [X_init]
        all_F = [F_init]
        
        used = init_count
        
        # Phase 2: Local Optimization with Weighted Sums
        # Define weight vectors covering the simplex for 3 objectives
        weights = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1/3.0, 1/3.0, 1/3.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5]
        ])
        
        n_weights = len(weights)
        remaining_budget = self.budget - used
        maxfun_per_run = max(10, int(remaining_budget / n_weights))
        
        for w in weights:
            if time.time() > deadline:
                break
            if used >= self.budget:
                break
            
            # Objective function for scalarization
            def obj_func(x, w=w):
                f = problem(x)
                # Clip to handle any numerical issues if problem doesn't clip internally
                # But problem(x) is supposed to return valid values.
                return np.dot(f, w)
            
            # Find the best starting point from initialization for this weight
            # We minimize the weighted sum
            weighted_sums = F_init @ w
            best_idx = np.argmin(weighted_sums)
            x0 = X_init[best_idx]
            
            # Run L-BFGS-B
            # maxfun limits the number of function evaluations
            # Note: L-BFGS-B with numerical gradients uses ~2*dim+1 evals per iteration
            # So maxfun=25 is approx 2 iterations.
            res = minimize(
                obj_func, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxfun': maxfun_per_run, 'maxiter': maxfun_per_run}
            )
            
            if res.success or res.status == 1: # 1 means maxfun reached
                # Record the final point
                x_res = res.x
                f_res = problem(x_res)
                
                # Check if this point is better than the start
                # We just record it. The hypervolume calculation will handle dominance.
                all_X.append(x_res.reshape(1, -1))
                all_F.append(f_res.reshape(1, -1))
                
                # Update used count
                # We don't know exactly how many evals scipy used, 
                # but we can estimate or just track if we exceed budget.
                # To be safe, we assume it used up to maxfun.
                used += maxfun_per_run
                
        # Concatenate results
        X_final = np.vstack(all_X)
        F_final = np.vstack(all_F)
        
        return X_final
