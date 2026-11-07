"""Multi-objective optimizer: Tchebycheff scalarization with Nelder-Mead local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 40s safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        used = 0
        all_F = []
        all_X = []
        
        def evaluate(x):
            nonlocal used
            if used >= budget:
                return None
            if time.time() > deadline:
                return None
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
            used += 1
            return f
        
        # Phase 1: Initial sampling to estimate ideal point
        # Use 10 points
        n_init = 10
        for _ in range(n_init):
            x = lo + np.random.random(dim) * (hi - lo)
            f = evaluate(x)
            if f is None:
                break
        
        if not all_F:
            return all_X
        
        # Estimate ideal point (minimum of each objective)
        F_init = np.array(all_F)
        ideal = np.min(F_init, axis=0)
        
        # Phase 2: Tchebycheff scalarization with Nelder-Mead
        # Generate weight vectors on the (n_obj-1) simplex
        # For n_obj=3, we can use a grid on the simplex
        n_weights = 15
        weights = []
        # Generate points on the 2D simplex (lambda1 + lambda2 + lambda3 = 1, lambda_i >= 0)
        # We use a grid approach:
        n_grid_w = 4
        for i in range(n_grid_w + 1):
            for j in range(n_grid_w + 1 - i):
                k = n_grid_w - i - j
                w = np.array([i, j, k], dtype=float)
                w = w / n_grid_w
                weights.append(w)
        
        # Ensure we don't have too many weights for the budget
        evals_per_run = max(5, (budget - n_init - 5) // len(weights))
        
        # For each weight, run Nelder-Mead
        # We need to map the problem to a scalar objective
        # Tchebycheff: min max_i (lambda_i * (f_i(x) - ideal_i))
        # Note: Since we minimize, and f_i are to be minimized, this is correct.
        # We use the ideal point as the reference.
        
        for w in weights:
            if used >= budget or time.time() > deadline:
                break
            
            def tcheb_obj(x):
                f = problem(x)
                # We count this evaluation
                nonlocal used
                if used >= budget:
                    return 1e10
                if time.time() > deadline:
                    return 1e10
                all_X.append(x)
                all_F.append(f)
                used += 1
                
                # Tchebycheff aggregation
                diff = f - ideal
                # Ensure non-negative for max? No, Tchebycheff usually uses absolute or just the value if ideal is known.
                # Standard Tchebycheff: max_i (lambda_i * (f_i(x) - z_i^*))
                # Since z^* is the minimum, f_i - z_i^* >= 0.
                vals = w * diff
                return np.max(vals)
            
            # Initial guess for Nelder-Mead
            # Use the point from initial sampling that minimizes the weighted sum for this weight
            # Or just a random point
            # Let's try to use a point that is good for this specific weight from the initial set
            if len(all_F) > 0:
                F_arr = np.array(all_F)
                weighted_sums = F_arr @ w
                best_init_idx = np.argmin(weighted_sums)
                x0 = np.array(all_X[best_init_idx])
            else:
                x0 = lo + 0.5 * (hi - lo)
            
            # Run Nelder-Mead
            # We need to be careful with the evaluation count.
            # scipy minimize will call tcheb_obj multiple times.
            # We set a limit on iterations and function evaluations.
            # However, scipy doesn't have a strict 'max_evals' that stops exactly, but 'maxiter' helps.
            # We rely on the budget check inside tcheb_obj.
            
            try:
                res = minimize(tcheb_obj, x0, method='Nelder-Mead', 
                               options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
            except Exception:
                pass
        
        return all_X
# EVOLVE-BLOCK-END
