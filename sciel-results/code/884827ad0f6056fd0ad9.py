"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Nelder-Mead with multi-start grid."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        # Create a small grid of initial points to handle rotation/shift
        # 5 dimensions. A 3x3x3x3x3 grid is 243 points.
        # Maybe 2x2x2x2x2 = 32 points is enough for multi-start.
        # Let's use a set of points covering the space.
        # Since the minimum is in [-4, 4]^5, we can sample there.
        
        # To be efficient, let's use a stratified sample or a coarse grid.
        # 3 points per dim -> 3^5 = 243 starts.
        # Each Nelder-Mead run uses ~100-200 calls.
        # 243 * 200 = 48600 calls. Way over budget of 500.
        
        # So we must limit the number of starts or the calls per start.
        # Budget is 500 total.
        # If we do 5 starts, we have 100 calls per start.
        # If we do 10 starts, we have 50 calls per start.
        
        # Let's try 5 starts at vertices of a hypercube in [-4, 4]^5?
        # Or just center + random?
        # The problem says "hidden seeded shift, rotation and offset".
        # This implies the landscape is a transformed Rosenbrock.
        # Rosenbrock has a narrow valley.
        
        # Strategy:
        # 1. Use a small set of initial points (e.g., 3-5).
        # 2. Run Nelder-Mead with a maxfun limit proportional to remaining budget.
        # 3. Keep the best result.
        
        # Let's use 3 starts:
        # x0_1: [0, 0, 0, 0, 0]
        # x0_2: [-2, -2, -2, -2, -2]
        # x0_3: [2, 2, 2, 2, 2]
        
        # Actually, since it's rotated, center might not be close.
        # Let's use a random seed for the optimizer to pick starts?
        # No, deterministic is better for reproducibility if seeds are fixed per instance?
        # The evaluator creates Optimizer(budget=500, dim=5) for each instance.
        # It doesn't say the seed is fixed across instances, but usually "seeded" implies reproducibility.
        # However, we don't know the seed.
        
        # Let's use a fixed set of diverse points.
        # Points:
        # 0: [0, 0, 0, 0, 0]
        # 1: [-3, -3, -3, -3, -3]
        # 2: [3, 3, 3, 3, 3]
        # 3: [-3, 3, -3, 3, -3]
        # 4: [3, -3, 3, -3, 3]
        
        # Total budget 500.
        # 5 starts -> 100 calls each.
        # Nelder-Mead is fast.
        
        starts = [
            np.zeros(self.dim),
            -3 * np.ones(self.dim),
            3 * np.ones(self.dim),
            np.array([-3, 3, -3, 3, -3]),
            np.array([3, -3, 3, -3, 3])
        ]
        
        best_x = None
        best_f = float('inf')
        
        calls_made = 0
        calls_per_start = max(10, (self.budget - len(starts)) // len(starts))
        
        for x0 in starts:
            if time.time() > deadline:
                break
            if calls_made >= self.budget:
                break
                
            # Clip start point to box
            x0 = np.clip(x0, -5, 5)
            
            # Wrapper to count calls and enforce deadline/budget
            local_calls = [0]
            local_best_x = [x0]
            local_best_f = [func(x0)]
            local_calls[0] += 1
            calls_made += 1
            
            def local_func(x):
                nonlocal local_calls, local_best_x, local_best_f, calls_made
                if local_calls[0] >= calls_per_start or time.time() > deadline:
                    return local_best_f[0] # Return current best to stop optimization effectively? 
                                           # Or raise exception? 
                                           # Better to return a large value to signal stop?
                                           # Actually, scipy minimize checks convergence.
                                           # If we return the same value, it might think it converged.
                                           # Let's just let it run but check time in loop?
                                           # No, we can't intercept scipy's internal loop easily without subclassing.
                                           # Let's rely on maxfun.
                                           # But maxfun doesn't stop if time runs out.
                                           # We need a hard stop.
                                           # If time runs out, we should break the outer loop.
                                           # But we are inside scipy.
                                           # We can raise an exception to break out of scipy?
                                           # Yes, raising an exception will stop minimize.
                                           # We catch it and use the last best value found by our wrapper.
                                   
                x = np.clip(x, -5, 5)
                local_calls[0] += 1
                calls_made += 1
                fx = func(x)
                if fx < local_best_f[0]:
                    local_best_f[0] = fx
                    local_best_x[0] = x
                return fx

            try:
                res = minimize(local_func, x0, method='Nelder-Mead', 
                               options={'maxiter': calls_per_start, 'xatol': 1e-6, 'fatol': 1e-6})
                # Note: maxiter in Nelder-Mead is not exactly maxfun, but related.
                # To be safe, we use the call counter in local_func.
                
                # Update global best
                if local_best_f[0] < best_f:
                    best_f = local_best_f[0]
                    best_x = local_best_x[0]
                    
            except Exception:
                # If time limit or other error, use what we have
                if local_best_f[0] < best_f:
                    best_f = local_best_f[0]
                    best_x = local_best_x[0]
                
        return best_x, best_f
# EVOLVE-BLOCK-END
