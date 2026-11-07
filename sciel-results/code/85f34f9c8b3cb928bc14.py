import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0 # 300s limit, keep 20s margin for safety
        
        rng = np.random.default_rng(0)
        calls_used = 0
        best_x = None
        best_f = np.inf
        
        def safe_call(x):
            nonlocal calls_used, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError()
            if calls_used >= self.budget:
                raise ValueError("Budget exceeded")
            
            val = func(x)
            calls_used += 1
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
                
            return val

        # Phase 1: Global Exploration with CMA-ES
        # CMA-ES is generally more robust than Nelder-Mead for 5D problems, 
        # especially with hidden rotations.
        # We run a few short CMA-ES instances to find a good basin.
        
        # Initial population size for CMA-ES
        # sigma0 is initial step size
        # maxfevals is the max function evaluations for this run
        
        # Let's try 3 restarts of CMA-ES with a moderate budget each
        n_restarts = 3
        calls_per_restart = 100 # 3 * 100 = 300 calls, leaving 200 for polish
        
        for i in range(n_restarts):
            if time.time() > deadline - 50: # Leave time for polish
                break
            if calls_used >= self.budget - 100: # Leave 100 for polish
                break
                
            x0 = rng.uniform(-5, 5, self.dim)
            
            try:
                # Use CMA-ES via scipy minimize
                # 'maxfevals' controls the number of function evaluations
                # We set it to calls_per_restart
                res = minimize(safe_call, x0, method='CMA-ES', 
                               options={
                                   'maxfevals': calls_per_restart,
                                   'popsize': 8, # Default is 4 + 2*ceil(3*ln(N)), for N=5 it's ~15. 8 is smaller to save calls
                                   'seed': i,
                                   'verbose': 0
                               })
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except (TimeoutError, ValueError):
                break
            except Exception:
                # If CMA-ES fails for other reasons, break and try polish
                break

        # Phase 2: Local Refinement with L-BFGS-B
        # L-BFGS-B is excellent for smooth functions like Rosenbrock once in the basin.
        # It uses gradient information (approximated) and can converge very fast.
        
        if best_x is not None and time.time() < deadline - 10 and calls_used < self.budget - 10:
            try:
                # L-BFGS-B requires bounds
                bounds = [(-5.0, 5.0)] * self.dim
                
                # maxiter is number of iterations, not function evals directly.
                # Each iteration can use multiple function evals for gradient estimation.
                # We want to use up the remaining budget.
                # Let's just set a high maxiter and rely on the time/budget guard in safe_call.
                # However, scipy's minimize doesn't have a maxfevals option for L-BFGS-B directly in all versions,
                # but we can control it via the safe_call wrapper which raises exceptions.
                
                # Actually, L-BFGS-B in scipy has 'maxiter' and 'ftol', 'gtol'.
                # It doesn't have a direct 'maxfevals' stop condition in the options dict for the method itself 
                # in the same way CMA-ES does. But our safe_call will raise an exception if budget is hit.
                
                res = minimize(safe_call, best_x, method='L-BFGS-B', 
                               bounds=bounds,
                               options={
                                   'maxiter': 1000, # High enough to let it run until budget/time limit
                                   'ftol': 1e-15,
                                   'gtol': 1e-15
                               })
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except (TimeoutError, ValueError):
                pass
            except Exception:
                pass

        return best_x, best_f
