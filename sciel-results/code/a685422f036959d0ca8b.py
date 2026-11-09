"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        call_count = 0
        
        # Wrapper to count calls and enforce budget
        def counted_func(x):
            nonlocal call_count
            if call_count >= budget:
                return best_f # Return a safe value to stop optimization if budget exceeded, 
                              # though L-BFGS-B usually stops via maxiter
            call_count += 1
            try:
                return func(x)
            except:
                return 1e100

        bounds = [(-5, 5)] * dim
        rng = np.random.default_rng(42)
        
        # Strategy: Multi-start L-BFGS-B
        # L-BFGS-B uses finite differences for gradients, which is efficient for smooth functions
        # like Rosenbrock, especially in the valley.
        # We perform multiple restarts to escape local minima or poor initializations.
        
        # Estimate cost: L-BFGS-B with finite differences in dim=5 uses ~5-10 evals per step.
        # maxiter=100 might use ~500 evals if it doesn't converge early.
        # We should limit maxiter per restart to ensure we do enough restarts.
        
        # Let's try 10 restarts.
        # If we use maxiter=20, that's ~100-200 evals per run. 10 runs = 1000-2000 evals. Too many.
        # We need to be stricter.
        
        # Let's monitor call_count.
        # If we do 5 restarts, each with maxiter=30, that's ~150-300 evals per run. Total 750-1500.
        # Still risky.
        
        # Better approach:
        # 1. Do a few quick random evaluations to get a baseline.
        # 2. Run L-BFGS-B from the best point found so far, with a small maxiter.
        # 3. Repeat.
        
        # Actually, L-BFGS-B is very good at polishing.
        # Let's do:
        # - 5 random starts.
        # - For each, run L-BFGS-B with maxiter=20 (approx 100 evals).
        # - If budget allows, run another round with tighter tolerances on the best point.
        
        num_restarts = 5
        
        for i in range(num_restarts):
            if time.time() > deadline:
                break
            if call_count >= budget - 50: # Reserve some budget for final polish
                break
            
            # Random start
            x0 = rng.uniform(-5, 5, dim)
            
            try:
                # L-BFGS-B with finite differences
                # maxiter=30 to keep cost low per restart
                res = minimize(counted_func, x0, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 30, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass
        
        # Final polish with the best point found
        if call_count < budget - 10 and time.time() < deadline:
            try:
                # Use the best_x found so far as the start point for a more precise run
                res = minimize(counted_func, best_x, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        return best_x, best_f

# EVOLVE-BLOCK-END
