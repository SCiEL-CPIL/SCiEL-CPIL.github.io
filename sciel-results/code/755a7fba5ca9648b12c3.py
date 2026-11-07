"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-Start Basin-Hopping Optimizer.
    
    Strategy:
    1. Divide the budget among multiple independent runs (islands).
    2. Each run starts from a random point.
    3. Each run uses a local optimization method (Nelder-Mead) to refine the starting point.
    4. This approach is robust against multimodality (like Lunacek bi-Rastrigin) 
       by exploring different basins of attraction in parallel (sequentially within the budget).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Deadline guard
        start_time = time.time()
        max_time = 300.0
        
        rng = np.random.default_rng(12345)
        
        best_x = None
        best_f = float('inf')
        
        # Track calls made
        calls_made = 0
        
        # We will perform multiple independent local optimizations.
        # How many? 
        # Each Nelder-Mead run in 5D typically uses ~50-200 evaluations to converge reasonably.
        # With 500 budget, we can afford roughly 5-10 restarts.
        # Let's aim for 10 restarts to cover the search space well.
        
        num_restarts = 10
        budget_per_restart = budget // num_restarts
        
        for i in range(num_restarts):
            # Check time and budget
            if time.time() - start_time > max_time - 5:
                break
            if calls_made >= budget:
                break
            
            # Random start point in [-5, 5]^dim
            x0 = rng.uniform(-5, 5, size=dim)
            
            # We need to count function calls inside the local optimizer.
            # SciPy's minimize does not expose a call counter directly in a way 
            # that easily integrates with a global budget without wrapping.
            # We wrap func to count calls.
            
            local_calls = [0]
            
            def wrapped_func(x):
                nonlocal calls_made
                # Check global budget and time inside the wrapper
                if calls_made >= budget:
                    return 1e10 # Return a bad value to stop optimization
                if time.time() - start_time > max_time - 5:
                    return 1e10
                
                local_calls[0] += 1
                calls_made += 1
                
                # Ensure x is a numpy array and clipped
                x = np.clip(np.asarray(x), -5, 5)
                return float(func(x))
            
            # Run Nelder-Mead
            # maxiter is a proxy for function calls, but we rely on the wrapper for hard limits.
            # For 5D, Nelder-Mead usually needs a few hundred evals for high precision, 
            # but we only have ~50 per restart. 
            # Let's use a simpler local search or limit Nelder-Mead iterations strictly.
            # Actually, with only ~50 calls per restart, Nelder-Mead might be too expensive 
            # if it doesn't converge fast. 
            # Alternative: Use a simple gradient-free local search or just rely on 
            # multiple random starts + one coarse local improvement.
            
            # Let's try Nelder-Mead with a low maxiter to see if it improves the start.
            # If it doesn't improve, the random start itself is the result.
            
            try:
                res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                               options={'maxiter': 20, 'xatol': 0.1, 'fatol': 0.1})
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                # If optimization fails, use the initial point's value if we evaluated it?
                # wrapped_func evaluates x0 as part of the simplex initialization usually.
                # But if it threw, we might have wasted calls.
                pass

        # If we still have budget left, do some random sampling or a final coarse search?
        # The current strategy uses all budget in the loop.
        
        if best_x is None:
            # Fallback: should not happen if budget > 0
            best_x = np.zeros(dim)
            best_f = 0.0 # Unknown, but we must return something
            
        return best_x, best_f
# EVOLVE-BLOCK-END
