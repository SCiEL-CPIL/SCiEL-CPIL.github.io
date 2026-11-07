import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # The evaluator kills at 330s. We aim to finish well before.
        # 300 calls is small, L-BFGS-B is fast.
        # Let's set a generous internal limit to avoid timeout, e.g., 290s.
        time_limit = 290.0 
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)

        # Internal function to track calls and best
        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            # Clip to box as per task description
            x = np.clip(x, -5.0, 5.0)
            try:
                f = float(func(x))
            except Exception:
                return np.inf
            
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Strategy: Interleaved Global/Local Search
        # Each cycle:
        # 1. Sample K random points
        # 2. Run L-BFGS-B from the best of those K points
        # This balances exploration and exploitation.
        
        # Budget 300.
        # Let's use K=5 random, M=45 local. Total 50 per cycle.
        # 6 cycles exactly.
        
        K_random = 5
        M_local = 45
        cycle_size = K_random + M_local
        
        # If budget is not divisible, the last cycle will be truncated by the while condition.
        
        while calls_made < self.budget:
            # Time check
            if time.time() - start_time > time_limit:
                break
            
            # 1. Global Sampling Phase
            current_best_x = None
            current_best_f = np.inf
            
            for _ in range(K_random):
                if calls_made >= self.budget:
                    break
                if time.time() - start_time > time_limit:
                    break
                
                x = rng.uniform(-5.0, 5.0, self.dim)
                f = safe_func(x)
                
                if f < current_best_f:
                    current_best_f = f
                    current_best_x = x.copy()
            
            if current_best_x is None:
                break

            # 2. Local Optimization Phase
            remaining_budget = self.budget - calls_made
            if remaining_budget <= 0:
                break
            
            # Cap local budget to remaining
            max_fev = min(M_local, remaining_budget)
            
            try:
                # L-BFGS-B is suitable for bound-constrained problems
                # We pass the bounds explicitly
                bounds = [(-5.0, 5.0)] * self.dim
                
                # Use a tight tolerance to ensure we use the budget if possible
                # or stop early if converged
                res = minimize(
                    safe_func,
                    current_best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxfev': max_fev,
                        'ftol': 1e-12,
                        'gtol': 1e-12
                    }
                )
            except Exception:
                # If minimization fails, we just continue with next cycle
                pass

        # Return the best found
        if best_x is None:
            # Fallback if no calls were made (should not happen)
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            
        return best_x, best_f
