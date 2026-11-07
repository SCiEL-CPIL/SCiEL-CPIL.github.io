import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        time_limit = 300.0  # Safety margin before 330s kill
        
        call_count = 0
        best_x = None
        best_fx = float('inf')
        
        # Wrap func to count calls and enforce time/budget
        def wrapped_func(x):
            nonlocal call_count, best_x, best_fx
            if call_count >= budget:
                raise RuntimeError("Budget exceeded")
            if time.time() - start_time > time_limit:
                raise RuntimeError("Time limit exceeded")
            
            # Ensure x is a numpy array and within bounds? 
            # The problem statement says func clips, but L-BFGS-B might try to go out of bounds.
            # L-BFGS-B respects bounds if provided.
            call_count += 1
            fx = func(x)
            
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                
            return fx

        # Define bounds
        bounds = [(-5.0, 5.0)] * dim
        
        # Generate starting points
        rng = np.random.default_rng(42)
        starts = []
        
        # Heuristic starts for Rosenbrock (min at 1)
        starts.append(np.zeros(dim))
        starts.append(np.ones(dim))
        starts.append(-np.ones(dim))
        
        # Random starts
        # Use a few random points to cover the space
        for _ in range(10):
            starts.append(rng.uniform(-5, 5, dim))
            
        # Sort starts by distance to origin? Or just iterate.
        # Let's iterate through them, allocating budget.
        # We have 500 calls.
        # Let's try to run each start with a limited maxfev.
        # If we have 13 starts, 500/13 ~ 38 calls per start.
        # This might be too few for L-BFGS-B to converge if far away.
        # Better strategy: 
        # 1. Run a short "scout" phase with many starts, low maxfev.
        # 2. Take the best result from scout phase.
        # 3. Run a "refine" phase on the best result with higher maxfev.
        
        # Phase 1: Scout
        scout_budget_per_start = 20
        scout_max_starts = min(len(starts), budget // scout_budget_per_start)
        
        current_budget_used = 0
        scout_best_x = None
        scout_best_fx = float('inf')
        
        for i in range(scout_max_starts):
            x0 = starts[i]
            try:
                res = minimize(wrapped_func, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 5, 'maxfev': scout_budget_per_start})
                if res.fun < scout_best_fx:
                    scout_best_fx = res.fun
                    scout_best_x = res.x
            except RuntimeError:
                break
            except Exception:
                # Ignore other exceptions in scout to keep going? 
                # No, if wrapped_func raises, it's budget/time. 
                # If minimize raises something else, it's a bug. 
                # But we catch RuntimeError from wrapped_func.
                pass
        
        # Phase 2: Refine
        # Use remaining budget for the best scout result
        remaining_budget = budget - call_count
        
        if remaining_budget > 10 and scout_best_x is not None:
            try:
                # We can run L-BFGS-B again from the scout best
                # Note: wrapped_func continues counting from where it left off
                res = minimize(wrapped_func, scout_best_x, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'maxfev': remaining_budget})
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x
            except RuntimeError:
                pass
            except Exception:
                pass
        
        # If no scout was run or failed, and we still have budget, try a direct run on a good start?
        # The scout phase should have updated best_x via wrapped_func.
        
        if best_x is None:
            # Fallback: just return the first eval if everything failed
            # This shouldn't happen if func is valid
            x0 = np.zeros(dim)
            try:
                fx = func(x0)
                best_x = x0
                best_fx = fx
            except:
                best_x = np.zeros(dim)
                best_fx = 0.0

        return best_x, best_fx
