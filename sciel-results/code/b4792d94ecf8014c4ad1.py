import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        n = self.dim
        budget = self.budget
        
        calls = 0
        best_f = np.inf
        best_x = None
        
        def wrapped_func(x):
            nonlocal calls
            if calls >= budget:
                return 1e10
            calls += 1
            # func clips internally per contract, but we clip here to be safe for our own logic
            # though the contract says func clips. 
            # "func(x) clips a finite length-5 vector to the box and returns a float."
            # So we can just pass x.
            return func(x)
        
        # Deterministic initial points to ensure reproducibility and coverage
        # We use a grid-like or Halton-like distribution for initial starts
        # For 5D, let's pick 10 diverse starting points.
        # Simple approach: random with fixed seed, or specific points.
        # Given the "hidden seeded shift, rotation and offset", random uniform is safe.
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Multi-start local optimization
        # We want to use the budget efficiently.
        # Nelder-Mead in 5D is efficient.
        # Let's do 10 restarts.
        # Each restart gets a limited budget of calls.
        
        num_restarts = 10
        calls_per_restart = 30  # Roughly 300 calls total
        
        for i in range(num_restarts):
            if calls >= budget or time.time() > deadline - 10:
                break
            
            # Generate a random start point in [-5, 5]^n
            x0 = rng.uniform(-5, 5, n)
            
            # Run Nelder-Mead
            try:
                res = scipy.optimize.minimize(
                    wrapped_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': calls_per_restart,
                        'xatol': 1e-4,
                        'fatol': 1e-4,
                        'adaptive': True
                    }
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass

        # Phase 2: Final polish on the best candidate
        if best_x is not None and calls < budget and time.time() < deadline - 5:
            remaining_budget = budget - calls
            # Use remaining budget for a more thorough Nelder-Mead on the best point
            # Cap maxiter to ensure we don't waste time, but allow it to converge
            max_iter_final = min(remaining_budget, 200)
            
            try:
                res = scipy.optimize.minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter_final,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass

        # Return interface must match current program: return best_x, best_f
        return best_x, best_f
