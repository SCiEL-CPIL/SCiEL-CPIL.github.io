"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Global search via L-BFGS-B from multiple random restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Count function evaluations
        call_count = [0]

        def wrapped_func(x):
            call_count[0] += 1
            if call_count[0] > budget:
                # Return a large value to stop optimization, though the loop below will also break
                return 1e10
            return func(x)

        best_x = None
        best_f = np.inf

        # We'll use a mix of restarts.
        # L-BFGS-B is a derivative-free (numerical gradient) method but very efficient.
        # We need to be careful with the budget.
        
        # Strategy: 
        # 1. Start with a few random points.
        # 2. Run L-BFGS-B from each, stopping if budget runs out.
        # 3. If budget remains, do more restarts or a local search.

        # To be safe and simple, let's just do a loop of restarts.
        # Each restart consumes some budget.
        
        # Estimate: L-BFGS-B in 5D might take ~10-50 evals to converge to a local min.
        # So we can afford ~10-50 restarts.
        
        # Let's try 20 restarts initially.
        num_restarts = 20
        
        for i in range(num_restarts):
            if call_count[0] >= budget:
                break
            
            # Random start
            x0 = rng.uniform(-5, 5, dim)
            
            # Run L-BFGS-B
            # bounds is a list of (min, max) for each dimension
            bounds = [(-5, 5)] * dim
            
            try:
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 100, 'maxls': 50}
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
                    
            except Exception:
                pass
            
            # If we are close to budget, stop
            if call_count[0] >= budget - 10:
                break

        # If we still have budget, do a fine-grained local search around the best point
        # using a simple hill climbing or just more L-BFGS-B restarts near the best
        if best_x is not None and call_count[0] < budget:
            # Try a few more restarts very close to the best solution
            for _ in range(10):
                if call_count[0] >= budget:
                    break
                
                # Small perturbation
                noise = rng.normal(0, 0.1, dim)
                x0 = np.clip(best_x + noise, -5, 5)
                
                try:
                    res = minimize(
                        wrapped_func, 
                        x0, 
                        method='L-BFGS-B', 
                        bounds=bounds,
                        options={'maxiter': 50, 'maxls': 20}
                    )
                    
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                        
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
