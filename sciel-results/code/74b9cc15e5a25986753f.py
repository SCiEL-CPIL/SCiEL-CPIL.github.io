import numpy as np
import time
from scipy.optimize import minimize

class BudgetExceeded(Exception):
    pass

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_fx = np.inf
        calls = 0

        def wrapped_func(x):
            nonlocal calls, best_x, best_fx
            if time.time() > deadline:
                raise BudgetExceeded("Time limit exceeded")
            if calls >= budget:
                raise BudgetExceeded("Budget exceeded")
            
            fx = func(x)
            calls += 1
            
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            
            return fx

        # Phase 1: Nelder-Mead to get a good starting point and handle rotation
        # NM is derivative-free and robust to scaling/rotation.
        # Use a small budget for this phase.
        nm_budget = min(50, budget // 4)
        
        x0 = np.zeros(dim)
        
        try:
            # Nelder-Mead
            res_nm = minimize(wrapped_func, x0, method='Nelder-Mead', 
                              options={'maxiter': 20, 'xatol': 1e-8, 'fatol': 1e-8})
        except BudgetExceeded:
            pass
        except Exception:
            pass

        # Phase 2: L-BFGS-B for precise convergence
        if calls < budget and time.time() < deadline:
            # Use the best point found so far as start for L-BFGS-B
            x_start = best_x if best_x is not None else x0
            
            try:
                res_lbfgs = minimize(wrapped_func, x_start, method='L-BFGS-B', 
                                     bounds=[(-5, 5) for _ in range(dim)], 
                                     options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            except BudgetExceeded:
                pass
            except Exception:
                pass

        # Phase 3: If we have significant budget left and haven't converged, 
        # try another L-BFGS-B from a slightly perturbed point or just continue?
        # L-BFGS-B usually converges in < 100 calls for d=5 if started well.
        # If NM used 50, L-BFGS-B uses ~50-100. Total ~150.
        # We have 500. We can afford a restart if needed.
        
        if calls < budget - 20 and time.time() < deadline:
             # Check if we are stuck or just finished.
             # If best_fx is still high, try a random restart.
             # But for convex functions, we should be close.
             # Let's do one more L-BFGS-B from the current best to ensure convergence.
             x_start = best_x if best_x is not None else x0
             try:
                 res_lbfgs2 = minimize(wrapped_func, x_start, method='L-BFGS-B', 
                                       bounds=[(-5, 5) for _ in range(dim)], 
                                       options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10})
             except BudgetExceeded:
                 pass
             except Exception:
                 pass

        return best_x, best_fx
