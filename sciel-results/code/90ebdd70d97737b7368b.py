"""Rewritten optimizer for BBOB f23 (Katsuura) using Multi-Start Nelder-Mead with ES polish."""
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
        deadline = start_time + 280.0 # 20s safety margin for 300s limit
        
        d = self.dim
        B = self.budget
        
        # State
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        def wrapper(x):
            nonlocal calls_made, best_x, best_f
            # Check time and budget
            if time.time() > deadline or calls_made >= B:
                return best_f if best_f != np.inf else 0.0 # Return a dummy value to stop optimizer
            
            # Clip to box [-5, 5]
            x_clip = np.clip(x, -5, 5)
            
            # Evaluate
            f_val = func(x_clip)
            calls_made += 1
            
            # Update best
            if f_val < best_f:
                best_f = f_val
                best_x = x_clip.copy()
                
            return f_val

        # 1. Multi-Start Strategy
        # Generate diverse starting points
        rng = np.random.default_rng(42)
        
        # Points to try:
        # 1. Origin
        # 2. Random in [-4, 4]^5
        # 3. Random in [-2, 2]^5
        # 4. Random in [-4, 4]^5
        # 5. Random in [-4, 4]^5
        
        start_points = []
        start_points.append(np.zeros(d))
        start_points.append(rng.uniform(-4, 4, d))
        start_points.append(rng.uniform(-2, 2, d))
        start_points.append(rng.uniform(-4, 4, d))
        start_points.append(rng.uniform(-4, 4, d))
        
        # Evaluate all start points first to find the best initial guess
        # This consumes 5 calls.
        for sp in start_points:
            if calls_made >= B or time.time() > deadline:
                break
            sp_clip = np.clip(sp, -5, 5)
            f_val = func(sp_clip)
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = sp_clip.copy()
        
        # 2. Local Refinement with Nelder-Mead
        # We have B - calls_made budget left.
        # We will try to refine the best point found so far, and potentially other good points if we have budget.
        # However, NM is local. It's better to refine the single best point found with the remaining budget,
        # or split the budget among the top few candidates.
        # Given the multimodal nature, refining the global best is usually the best use of remaining budget.
        
        # Let's allocate the remaining budget to Nelder-Mead on the current best_x.
        remaining_budget = B - calls_made
        if remaining_budget > 10:
            # Nelder-Mead options
            # maxiter is not exactly function evaluations, but a good proxy.
            # In 5D, NM takes roughly d+1 simplex points per iteration.
            # Let's set maxiter to a value that allows ~remaining_budget evaluations.
            max_iter = int(remaining_budget / 2) + 10
            
            try:
                res = minimize(
                    wrapper, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
            except Exception:
                pass

        # 3. Final Polish with (1+1)-ES
        # If we have budget left and time, do a quick ES polish.
        if calls_made < B and time.time() < deadline:
            x = best_x
            fx = best_f
            sigma = 0.1
            # Simple (1+1)-ES
            while calls_made < B and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.5
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.7
                    if sigma < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
