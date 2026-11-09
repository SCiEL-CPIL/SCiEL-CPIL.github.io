"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start Nelder-Mead optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        # Pre-allocate a counter to track calls
        calls = [0]
        max_calls = self.budget

        def safe_func(x):
            if calls[0] >= max_calls:
                return 1e10
            calls[0] += 1
            # func clips to [-5, 5] and returns float
            return func(x)

        best_x = np.zeros(self.dim)
        best_f = np.inf
        
        # Generate diverse starting points
        # 1. Origin
        # 2. A few deterministic shifts
        # 3. Random points (seeded for reproducibility if needed, but random is fine)
        rng = np.random.default_rng(42)
        
        start_points = [np.zeros(self.dim)]
        
        # Add some deterministic offsets
        for d in range(self.dim):
            for val in [-2.0, 2.0]:
                pt = np.zeros(self.dim)
                pt[d] = val
                start_points.append(pt)
        
        # Add random points
        # We have 500 calls. Nelder-Mead in 5D takes roughly (dim+1)*iterations calls.
        # With maxiter=50, it takes ~6*50 = 300 calls per run.
        # So we can afford ~1-2 full runs, or more if we stop early.
        # Let's use a smaller maxiter to allow more restarts.
        # If maxiter=20, calls ~ 6*20 = 120. We can do 4 restarts.
        
        n_restarts = 4
        for _ in range(n_restarts):
            start_points.append(rng.uniform(-4, 4, self.dim))
            
        # Sort start points by a heuristic? 
        # For Rosenbrock, the valley is important. 
        # The origin is a standard start. 
        # Random points might be better if the shift is large.
        
        # We will iterate through start points, running Nelder-Mead.
        # We stop a run if we hit the time limit or call limit.
        
        for start_pt in start_points:
            if time.time() > deadline:
                break
            if calls[0] >= max_calls:
                break
                
            # Nelder-Mead parameters
            # xatol and fatol help stop early if converged
            try:
                res = minimize(
                    safe_func, 
                    start_pt, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': 30,  # Keep it small to allow restarts
                        'xatol': 1e-4, 
                        'fatol': 1e-4,
                        'adaptive': True # Helps with scaling issues
                    }
                )
                
                if res.success or res.fun < best_f:
                    # Check if we actually made calls
                    if calls[0] > 0:
                        best_f = res.fun
                        best_x = res.x.copy()
                        
            except Exception:
                # If scipy fails for any reason, continue to next start point
                pass

        # If no valid calls were made (should not happen), return origin
        if best_f == np.inf:
             best_x = np.zeros(self.dim)
             best_f = func(best_x)
             calls[0] = 1

        return best_x, best_f
# EVOLVE-BLOCK-END
