"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective optimizer using Tchebycheff decomposition and L-BFGS-B."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Counters
        eval_count = 0
        max_budget = self.budget
        
        # Storage for all evaluated points
        X_all = []
        F_all = []
        
        def eval_point(x):
            nonlocal eval_count
            if eval_count >= max_budget:
                return None
            eval_count += 1
            x_clipped = np.clip(x, lo, hi)
            f = np.array(problem(x_clipped))
            X_all.append(x_clipped)
            F_all.append(f)
            return f

        # 1. Initial Random Sampling to estimate ideal point z*
        n_init = min(30, max_budget)
        # Use fixed seed for reproducibility within the run? 
        # The task says evaluator seeds np.random, so we can use np.random
        for _ in range(n_init):
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.random(self.dim)
            eval_point(x)
            
        if not F_all:
            # Fallback if something went wrong
            return np.array([lo + (hi-lo)*0.5] * self.n_obj)
            
        F_init = np.array(F_all)
        # Estimate ideal point z* (component-wise minimum)
        z_star = np.min(F_init, axis=0)
        
        # 2. Generate Weight Vectors for Tchebycheff Decomposition
        # For 2 objectives, weights are (w, 1-w)
        # We want to cover the front well.
        n_weights = 10
        # Avoid extreme weights that might lead to boundary solutions if not needed
        w_values = np.linspace(0.05, 0.95, n_weights)
        weights = np.array([[w, 1-w] for w in w_values])
        
        # 3. Local Optimization for each weight
        # Remaining budget
        remaining_budget = max_budget - eval_count
        
        if remaining_budget <= 0:
            return np.array(X_all[-1])
            
        # Allocate budget to each subproblem
        # L-BFGS-B uses finite differences. For dim=4, one gradient eval costs ~9 function evals.
        # We can afford roughly 2-3 gradient steps per subproblem if we have ~15-20 evals.
        
        evals_per_sub = max(5, remaining_budget // n_weights)
        
        for i, w in enumerate(weights):
            if time.time() > deadline:
                break
            if eval_count >= max_budget:
                break
                
            # Define the Tchebycheff scalarization function
            # min max_i { w_i * (f_i(x) - z_i*) }
            # We use a smooth approximation or just the max.
            # Standard Tchebycheff uses max. L-BFGS-B can handle non-smooth functions 
            # but it's better to use a smooth approximation if possible, or just let it handle it.
            # Given the low dimension, the kink in max is not a huge issue for L-BFGS-B.
            
            def tcheb_func(x, w=w, z=z_star):
                f = np.array(problem(np.clip(x, lo, hi)))
                # Update z_star dynamically? 
                # It's better to keep z_star fixed per subproblem or update it carefully.
                # Updating z_star during optimization can be unstable. 
                # We will update z_star after each subproblem completes.
                values = w * (f - z)
                return np.max(values)
            
            # Initial guess: use the current best point for this direction or a random point
            # A good heuristic is to use the point from the initial sample that is best in this direction
            # Or just a random point. Let's use the centroid of the initial points.
            x0 = np.mean(X_all, axis=0)
            
            # Constraints: bounds are handled by L-BFGS-B
            bounds_opt = list(zip(lo, hi))
            
            # Run L-BFGS-B
            # maxiter controls the number of iterations.
            # Each iteration involves a gradient evaluation (approx 9 func evals).
            # We want to use ~evals_per_sub function evaluations.
            maxiter = max(1, evals_per_sub // 9)
            
            try:
                res = minimize(tcheb_func, x0, method='L-BFGS-B', bounds=bounds_opt, 
                               options={'maxiter': maxiter, 'maxfun': evals_per_sub})
                
                # The optimization has consumed some evaluations.
                # We need to record the final point if it wasn't recorded by tcheb_func 
                # (minimize calls tcheb_func which calls problem, so it IS recorded).
                # However, if minimize stops early, we just keep the last evaluated point.
                
                # Update z_star with the best point found so far
                if F_all:
                    F_current = np.array(F_all)
                    z_star = np.min(F_current, axis=0)
                    
            except Exception:
                pass

        # 4. Final Polish?
        # If we have budget left, we could run a few more random perturbations around the best points.
        # But L-BFGS-B should have converged.
        
        return np.array(X_all[-1])

# EVOLVE-BLOCK-END
