"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer using Weighted Sum + L-BFGS-B."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Reference point for scaling
        ref = np.array([1705.0, 12.0, 0.3])
        
        X_archive = []
        F_archive = []
        eval_count = 0
        
        def safe_problem(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                # Return a very bad value to discourage further search in this direction
                # This value is not added to archive if we check before adding?
                # No, the task says "Every returned objective vector is recorded".
                # If we return a dummy, it gets recorded.
                # It's better to just stop calling problem.
                # But L-BFGS-B will call it.
                # So we must prevent L-BFGS-B from running if budget is exhausted.
                # We check budget before starting L-BFGS-B.
                # Inside L-BFGS-B, if we hit budget, we can't easily stop it.
                # However, 200 evals is small. L-BFGS-B is fast.
                # We will rely on maxiter to keep it within bounds.
                return [1e10, 1e10, 1e10]
            
            f = problem(x)
            eval_count += 1
            X_archive.append(x)
            F_archive.append(f)
            return f

        # 1. Initial Random Sampling
        n_init = 20
        for _ in range(n_init):
            if eval_count >= self.budget:
                break
            x = lo + (hi - lo) * np.random.rand(self.dim)
            safe_problem(x)
            
        if eval_count >= self.budget:
            return X_archive
            
        # 2. Generate Weight Vectors
        # Scale factors to balance objectives
        scale = ref
        weights = []
        
        # Extreme points
        for i in range(self.n_obj):
            w = np.zeros(self.n_obj)
            w[i] = 1.0
            weights.append(w)
            
        # Mid-edge points
        for i in range(self.n_obj):
            for j in range(self.n_obj):
                if i < j:
                    w = np.zeros(self.n_obj)
                    w[i] = 0.5
                    w[j] = 0.5
                    weights.append(w)
                    
        # Center point
        weights.append(np.ones(self.n_obj) / self.n_obj)
        
        # Add random weights for diversity
        for _ in range(5):
            weights.append(np.random.dirichlet(np.ones(self.n_obj)))
            
        np.random.shuffle(weights)
        
        # 3. Refinement with L-BFGS-B
        remaining_budget = self.budget - eval_count
        n_weights = len(weights)
        # Allocate budget per weight
        # We want to ensure we don't run out of budget in the middle of an optimization
        # L-BFGS-B uses ~1-2 function calls per iteration.
        # Let's reserve some budget for safety.
        safe_budget = remaining_budget - 5
        if safe_budget < 10:
             # If budget is too low, just do random search
             while eval_count < self.budget:
                 x = lo + (hi - lo) * np.random.rand(self.dim)
                 safe_problem(x)
             return X_archive

        budget_per_weight = max(5, safe_budget // n_weights)
        
        for w in weights:
            if eval_count >= self.budget - 2: # Keep 2 evals in reserve
                break
                
            # Effective weights for scaling
            w_eff = w / scale
            
            # Find best starting point in current archive for this direction
            if F_archive:
                F_arr = np.array(F_archive)
                # Use scaled objectives for comparison
                F_scaled = F_arr / scale
                scores = F_scaled @ w
                best_idx = np.argmin(scores)
                x0 = X_archive[best_idx].copy()
            else:
                x0 = (lo + hi) / 2.0
                
            def scalar_obj(x):
                f = safe_problem(x)
                # Use scaled objective for optimization
                return np.dot(w_eff, f)
                
            try:
                # Limit iterations to stay within budget
                # maxiter is number of iterations, not function calls.
                # Assume 2 function calls per iteration.
                max_iter = max(1, budget_per_weight // 2)
                
                res = minimize(scalar_obj, x0, method='L-BFGS-B', 
                               bounds=bounds, 
                               options={'maxiter': max_iter, 'maxls': 20})
                
                # The best point found by L-BFGS-B is res.x
                # It has already been evaluated in the last step of L-BFGS-B
                # and added to the archive by safe_problem.
                
            except Exception:
                pass

        # 4. Final Random Fill if budget remains
        while eval_count < self.budget:
            x = lo + (hi - lo) * np.random.rand(self.dim)
            safe_problem(x)

        return X_archive
# EVOLVE-BLOCK-END
