# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    DTLZ1-specific optimizer.
    
    Strategy:
    1. Use a small number of evaluations to find a point where g(x) is minimized 
       (convergence variables x[1:] at 0.5) using Nelder-Mead.
    2. Use the remaining budget to sample the diversity variable x[0] uniformly 
       over [0, 1] while keeping the convergence variables at the optimal point.
    This maximizes hypervolume by ensuring points lie exactly on the Pareto front 
    and are evenly distributed along it.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, keep 20s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        
        # DTLZ1 structure:
        # x[0] is the diversity variable (z_d)
        # x[1:] are the convergence variables (x_1...x_{d-1})
        
        # Initial guess for convergence variables: midpoint of bounds
        x_conv_init = 0.5 * (lo[1:] + hi[1:])
        
        # We need to find x_conv such that g(x_conv) is minimized.
        # g(x) = sum((x_i - 0.5)^2) for standard DTLZ1 with bounds [0,1].
        # The objective function problem(x) returns [f1, f2, f3].
        # f3 = 0.5 * g(x). So minimizing f3 is equivalent to minimizing g(x).
        # However, f1 and f2 also depend on x[0]. 
        # To isolate g, we can fix x[0] to a constant (e.g., 0.5) and minimize f3.
        
        x0_fixed = 0.5 * (lo[0] + hi[0])
        
        def objective_conv(x_conv):
            x = np.zeros(dim)
            x[0] = x0_fixed
            x[1:] = x_conv
            # Clip to bounds just in case
            x = np.clip(x, lo, hi)
            f = problem(x)
            # Return f3 which is proportional to g
            return f[2]

        used_evals = 0
        max_opt_evals = 50  # Limit evaluations for optimization
        
        # Use Nelder-Mead to minimize g
        # We need to track evaluations to respect budget
        eval_count = [0]
        
        def counting_objective(x_conv):
            if eval_count[0] >= max_opt_evals:
                return 1e10
            eval_count[0] += 1
            used_evals += 1
            return objective_conv(x_conv)
            
        # Perform optimization
        # Only if we have enough budget
        if self.budget > max_opt_evals + 10:
            try:
                # Nelder-Mead is derivative-free and robust
                res = minimize(counting_objective, x_conv_init, method='Nelder-Mead',
                               options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-12})
                x_conv_opt = res.x
            except Exception:
                x_conv_opt = x_conv_init
        else:
            x_conv_opt = x_conv_init
            # If budget is tight, just use the midpoint
            # We might not have called problem even once in optimization if budget was 0,
            # but problem requires at least 1 eval to be valid? No, it just records.
            # But we should probably do at least 1 eval to be safe?
            # The contract says "More than 200 evaluations... makes result invalid".
            # It doesn't say "less than 1 is invalid". But let's assume we should use the budget.
            
        # Clip optimized x_conv to bounds
        x_conv_opt = np.clip(x_conv_opt, lo[1:], hi[1:])
        
        # Now, spend the remaining budget sampling the diversity variable x[0]
        remaining_budget = self.budget - used_evals
        
        if remaining_budget <= 0:
            # We used all budget in optimization. Return whatever was recorded.
            # The problem function records calls. We don't need to return the array 
            # because the evaluator records the calls to problem().
            # Wait, the contract says "Every returned objective vector is recorded".
            # It implies the evaluator wraps problem or intercepts calls.
            # "The return value of __call__ is ignored."
            # So I don't need to return anything specific, just ensure problem is called.
            return None

        # Create a grid of points for x[0] to cover the Pareto front uniformly.
        # The Pareto front is a simplex in 3D. Uniform distribution in x[0] 
        # (the last variable in DTLZ1, which is x[0] in our 0-indexed array where x[0] is diversity)
        # Wait, standard DTLZ1:
        # f1 = 0.5 * g * (1 + x_{d-1})
        # f2 = 0.5 * g * (1 - x_{d-1})
        # f3 = 0.5 * g
        # Here x_{d-1} is the diversity variable. In 0-indexed numpy array of size dim,
        # if dim=7, x[6] is the diversity variable?
        # Let's check standard DTLZ1 definition.
        # Usually x = (x_1, ..., x_d).
        # g(x) = sum_{i=1}^{d-1} (x_i - 0.5)^2.
        # x_d is the diversity variable.
        # In my code above, I assumed x[0] is diversity and x[1:] are convergence.
        # Let's re-read the task description or standard convention.
        # "7 variables, bounds [0, 1]".
        # Usually DTLZ1 is defined such that the last variable is the diversity one.
        # Let's verify.
        # If x[0] is diversity, then g depends on x[1:].
        # If x[6] is diversity, then g depends on x[0:6].
        
        # Let's check the current program's assumption.
        # Current program:
        # center[:dim_conv] = 0.5 ... where dim_conv = dim - 1.
        # x[dim_conv] = ... (so x[6] is diversity).
        # So the current program assumes x[0..5] are convergence and x[6] is diversity.
        # My code above assumed x[0] is diversity and x[1:] are convergence.
        # I MUST FIX THIS.
        
        # Correct Assumption:
        # x[0] to x[dim-2] are convergence variables.
        # x[dim-1] is the diversity variable.
        
        # Let's restart the logic with correct indexing.
        
        # Reset state? No, I already called problem in optimization.
        # I need to discard the previous logic and write it correctly.
        
        # Since I can't "undo" the previous thought process in the code, I will write the final code carefully.
        
        # RE-IMPLEMENTATION:
        
        # 1. Identify convergence variables: x[0 : dim-1]
        # 2. Identify diversity variable: x[dim-1]
        
        # Optimization of convergence variables
        x_conv_init = 0.5 * (lo[:dim-1] + hi[:dim-1])
        x_div_fixed = 0.5 * (lo[dim-1] + hi[dim-1])
        
        def objective_conv_correct(x_conv):
            x = np.zeros(dim)
            x[:dim-1] = x_conv
            x[dim-1] = x_div_fixed
            x = np.clip(x, lo, hi)
            f = problem(x)
            return f[2] # f3 = 0.5 * g
            
        # We already used some evals in the "wrong" logic above?
        # No, this is a new code block. I will write the correct code from scratch.
        
        # The code below is the actual implementation.
        
        # --- CORRECT IMPLEMENTATION START ---
        
        # Reset counters for this specific call instance
        # (Note: The class is instantiated per seed, so state is fresh)
        
        used_evals = 0
        max_opt_evals = min(50, self.budget // 2) # Use half budget for optimization if possible
        
        if self.budget <= max_opt_evals:
             # If budget is very small, just do random search or midpoint
             x_conv_opt = 0.5 * (lo[:dim-1] + hi[:dim-1])
        else:
             # Optimize convergence variables
             x_conv_init = 0.5 * (lo[:dim-1] + hi[:dim-1])
             x_div_fixed = 0.5 * (lo[dim-1] + hi[dim-1])
             
             def counting_obj(x_conv):
                 nonlocal used_evals
                 if used_evals >= max_opt_evals:
                     return 1e10
                 used_evals += 1
                 x = np.zeros(dim)
                 x[:dim-1] = x_conv
                 x[dim-1] = x_div_fixed
                 x = np.clip(x, lo, hi)
                 f = problem(x)
                 return f[2]
                 
             try:
                 res = minimize(counting_obj, x_conv_init, method='Nelder-Mead',
                                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-12})
                 x_conv_opt = res.x
             except Exception:
                 x_conv_opt = x_conv_init
                 
        x_conv_opt = np.clip(x_conv_opt, lo[:dim-1], hi[:dim-1])
        
        remaining_budget = self.budget - used_evals
        
        if remaining_budget <= 0:
            return None
            
        # Sample diversity variable x[dim-1] uniformly
        # To maximize hypervolume, we want to cover the front [0, 1] for the diversity variable.
        # The Pareto front in DTLZ1 is a triangle in objective space.
        # Uniform sampling of the diversity variable provides good coverage.
        
        n_samples = remaining_budget
        
        # Use linspace for uniform coverage
        # Include endpoints 0 and 1 to maximize the extent of the front covered
        # However, at endpoints, two objectives become 0, which is good for hypervolume.
        
        z_values = np.linspace(0.0, 1.0, n_samples)
        
        for z_val in z_values:
            if used_evals >= self.budget:
                break
            if time.time() > deadline:
                break
                
            x = np.zeros(dim)
            x[:dim-1] = x_conv_opt
            x[dim-1] = lo[dim-1] + (hi[dim-1] - lo[dim-1]) * z_val
            x = np.clip(x, lo, hi)
            
            problem(x)
            used_evals += 1
            
        return None
        
# EVOLVE-BLOCK-END
