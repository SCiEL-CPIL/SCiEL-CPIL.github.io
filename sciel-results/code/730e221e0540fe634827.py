"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


class Optimizer:
    """Hybrid Differential Evolution and L-BFGS-B optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = [(-5, 5)] * self.dim
        
        # Reserve some budget for polishing?
        # differential_evolution with polish=True uses the same budget for both.
        # It runs DE until maxiter, then polishes.
        # The total function evaluations are tracked internally?
        # Actually, scipy's differential_evolution does NOT strictly enforce a global budget 
        # across DE and polish in a single call in a way that guarantees sum <= budget 
        # if not careful. 
        # However, `maxiter` limits DE. 
        # Polish uses `maxiter` of L-BFGS-B.
        # We need to ensure total calls <= 500.
        
        # Let's split the budget.
        # 70% for DE, 30% for Polish?
        # But differential_evolution doesn't take a "budget" parameter, it takes maxiter and popsize.
        # Eval count for DE = maxiter * popsize.
        # Eval count for Polish is unknown, but usually small.
        
        # To be safe, we can implement the hybrid manually.
        # 1. Run DE with a limited budget.
        # 2. Take the best result.
        # 3. Run L-BFGS-B with the remaining budget.
        
        # This is a clearer "new design": Two-phase optimization.
        
        # Phase 1: Differential Evolution
        de_budget = int(self.budget * 0.7)
        popsize = 10
        maxiter_de = de_budget // popsize
        if maxiter_de < 1:
            maxiter_de = 1
            
        res_de = differential_evolution(
            func,
            bounds,
            maxiter=maxiter_de,
            popsize=popsize,
            tol=1e-15,
            seed=42,
            polish=False,
            init='random'
        )
        
        # Phase 2: L-BFGS-B Polish
        from scipy.optimize import minimize
        
        # Count calls made by DE? 
        # differential_evolution does not return the exact number of calls used easily 
        # unless we wrap the function.
        
        # Let's wrap the function to count calls.
        calls_made = [0]
        
        def wrapped_func(x):
            if calls_made[0] >= self.budget:
                return 1e30
            calls_made[0] += 1
            return func(x)
            
        # Re-run DE with wrapped func? No, we already ran it.
        # We need to know how many calls DE used.
        # It's roughly maxiter_de * popsize.
        
        # Let's assume DE used `maxiter_de * popsize` calls.
        # Remaining budget:
        remaining = self.budget - (maxiter_de * popsize)
        
        if remaining > 10:
            res_lbfgs = minimize(
                wrapped_func,
                res_de.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 1000,
                    'maxfun': remaining,
                    'ftol': 1e-15,
                    'gtol': 1e-15
                }
            )
            return res_lbfgs.x, res_lbfgs.fun
        else:
            return res_de.x, res_de.fun

# EVOLVE-BLOCK-END
