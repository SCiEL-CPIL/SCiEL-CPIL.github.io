import numpy as np
import time

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False
    from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Deadline guard
        start_time = time.time()
        deadline = start_time + 290  # 10s margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Multi-start strategy with deterministic seeds and step sizes
        # We allocate the budget across multiple starts.
        # For Sphere, a single good start is often enough, but multi-start
        # with smaller steps helps if the first one gets stuck in a "plateau" 
        # due to numerical issues or if the shift is large (though CMA-ES handles shifts well).
        # Actually, for Sphere, CMA-ES is very robust. The main risk is using up budget 
        # without converging. 
        # Let's try a sequence of starts with different sigma0.
        
        starts = [
            {'x0': np.zeros(dim), 'sigma0': 4.0, 'seed': 42},
            {'x0': np.zeros(dim), 'sigma0': 1.0, 'seed': 123},
            {'x0': np.zeros(dim), 'sigma0': 0.2, 'seed': 999},
            {'x0': np.array([1.0, -1.0, 1.0, -1.0, 1.0]), 'sigma0': 2.0, 'seed': 555},
            {'x0': np.array([-2.0, 2.0, -2.0, 2.0, -2.0]), 'sigma0': 2.0, 'seed': 777},
        ]
        
        # Allocate budget: give most to the first, less to subsequent ones if they start
        # But CMA-ES stops when converged. 
        # We will run each start with a max of `budget` evaluations, but stop early if 
        # time is running out or if we find a very good solution.
        
        # To ensure we don't exceed budget, we track calls.
        # CMA-ES `maxfvals` is per run. We need to manage total calls.
        
        for start_info in starts:
            if time.time() > deadline - 5:
                break
            
            # If we already have a very good solution, we might skip further starts
            # But for Sphere, if we haven't found f_opt, we should keep trying.
            # However, CMA-ES on Sphere usually converges to machine precision quickly.
            # If the first one fails to converge, it's likely due to a bad start or 
            # the hidden shift making it look non-convex? No, Sphere is convex.
            # CMA-ES should find it.
            
            # Let's just run the next start if we haven't used all budget.
            # We'll give each start a portion of the budget.
            # Start 1: 60% of budget
            # Start 2: 30% of budget
            # Start 3: 10% of budget
            # This is a heuristic.
            
            # Actually, simpler: just run CMA-ES with maxfvals = remaining budget.
            # If it converges, it stops early. If not, it uses the budget.
            # Then we move to the next start? No, that would exceed budget.
            
            # We must track `calls_made`.
            # CMA-ES doesn't easily report exactly how many calls it made if we interrupt?
            # `res.evaluations` gives the number.
            
            if HAS_CMA:
                try:
                    # Use a wrapper to count calls? Or just trust cma's count.
                    # If we run cma.fmin, it uses up to maxfvals.
                    # We set maxfvals to the remaining budget.
                    remaining_budget = budget - calls_made
                    if remaining_budget <= 0:
                        break
                    
                    # We want to be careful. If the first run uses 100 calls and converges,
                    # we have 400 left. Should we run another start?
                    # For Sphere, if it converged, we are done.
                    # If it didn't converge (hit maxfvals), it might be stuck.
                    # A restart with different sigma might help.
                    
                    x0 = start_info['x0']
                    sigma0 = start_info['sigma0']
                    seed = start_info['seed']
                    
                    options = {
                        'bounds': (-5, 5),
                        'maxfvals': remaining_budget,
                        'seed': seed,
                        'tolfun': 1e-14,
                        'tolfunint': 1e-14,
                        'tolfunup': 1e-12,
                        'verbose': -9,
                        'bounds': (-5, 5)
                    }
                    
                    res = cma.fmin(func, x0, sigma0, options=options)
                    
                    # Update best
                    if res.fbest < best_f:
                        best_f = res.fbest
                        best_x = res.xbest
                    
                    # Update calls made
                    calls_made += res.evaluations
                    
                    # Check for convergence
                    if res.fbest < 1e-10: # Very close to optimum (assuming f_opt is near 0 or small)
                         # Actually f_opt is in [-100, 100]. We don't know it.
                         # But if the function value is very small, we are likely done.
                         # Let's break if we are very confident.
                         pass # Continue to be safe, or break if budget is low.
                         
                except Exception as e:
                    # If CMA-ES fails, fallback to Powell for remaining budget
                    if time.time() < deadline:
                        try:
                            remaining_budget = budget - calls_made
                            if remaining_budget > 0:
                                res = minimize(func, start_info['x0'], method='Powell', 
                                               options={'maxfev': remaining_budget})
                                if res.fun < best_f:
                                    best_f = res.fun
                                    best_x = res.x
                                # Estimate calls made by Powell. It's tricky.
                                # Assume it used all remaining budget?
                                # Or we can just stop.
                                calls_made = budget 
                        except Exception:
                            pass
            else:
                # Fallback to Powell
                try:
                    remaining_budget = budget - calls_made
                    if remaining_budget > 0:
                        res = minimize(func, start_info['x0'], method='Powell', 
                                       options={'maxfev': remaining_budget, 'xtol': 1e-12, 'ftol': 1e-12})
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = res.x
                        calls_made = budget
                except Exception:
                    pass

        return best_x, best_f
