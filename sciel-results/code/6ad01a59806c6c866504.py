"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Global search via Nelder-Mead from multiple random restarts."""

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

        # Nelder-Mead is a derivative-free method, robust to sharp ridges.
        # We use multiple restarts with diverse initial points.
        
        num_restarts = 6
        
        for i in range(num_restarts):
            if call_count[0] >= budget:
                break
            
            # Random start
            x0 = rng.uniform(-5, 5, dim)
            
            # Create a custom initial simplex with a reasonable step size
            # to ensure the simplex spans the domain adequately.
            step = 1.0
            simplex = [x0]
            for j in range(dim):
                x_new = x0.copy()
                x_new[j] += step
                simplex.append(x_new)
            
            try:
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='Nelder-Mead', 
                    options={
                        'xatol': 1e-6, 
                        'fatol': 1e-6, 
                        'maxiter': 100,
                        'initial_simplex': simplex
                    }
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
                    
            except Exception:
                pass
            
            # If we are close to budget, stop
            if call_count[0] >= budget - 20:
                break

        # If we still have budget, do a fine-grained local search around the best point
        # using a smaller initial step size for Nelder-Mead
        if best_x is not None and call_count[0] < budget:
            for _ in range(5):
                if call_count[0] >= budget:
                    break
                
                # Small perturbation or just restart from best with tighter tol
                x0 = best_x.copy()
                
                # Smaller step for refinement
                step = 0.1
                simplex = [x0]
                for j in range(dim):
                    x_new = x0.copy()
                    x_new[j] += step
                    simplex.append(x_new)

                try:
                    res = minimize(
                        wrapped_func, 
                        x0, 
                        method='Nelder-Mead', 
                        options={
                            'xatol': 1e-8, 
                            'fatol': 1e-8, 
                            'maxiter': 50,
                            'initial_simplex': simplex
                        }
                    )
                    
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                        
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
