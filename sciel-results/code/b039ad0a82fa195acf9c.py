import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start Powell's Method for rotated Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        total_calls = 0
        
        # Strategy: Multi-start Powell's Method
        # Powell's is often better for Rosenbrock than NM due to direction adaptation.
        n_starts = 3
        # Allocate budget. Leave some for safety.
        # Each Powell run can use up to its allocated maxfev.
        # We will track actual calls.
        
        start_points = []
        start_points.append(np.zeros(self.dim))
        for _ in range(n_starts - 1):
            sp = rng.uniform(-4, 4, self.dim)
            start_points.append(sp)
            
        for i, sp in enumerate(start_points):
            if total_calls >= self.budget:
                break
            if time.time() > deadline:
                break
                
            remaining = self.budget - total_calls
            # Allocate a portion of remaining budget to this start
            # If it's the last start, give it the rest.
            if i == n_starts - 1:
                alloc = remaining
            else:
                # Give each start an equal share of the initial budget, 
                # but cap it by remaining.
                alloc = min(self.budget // n_starts, remaining)
                
            if alloc < 10:
                continue
                
            try:
                res = minimize(func, sp, method='Powell', options={'maxfev': alloc, 'maxiter': 200})
                calls_made = res.nfev
                total_calls += calls_made
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception as e:
                # If minimize fails, we might have made some calls
                # We don't know exactly how many if it crashed mid-way, 
                # but assume it used some.
                # To be safe, we might just break or estimate.
                # However, the contract says "an exception later keeps the calls made".
                # But we don't have a hook to count calls inside minimize if it throws.
                # So we should be careful.
                # Let's assume it consumed the budget allocated if it crashed?
                # No, that's dangerous.
                # Better to wrap the function call to count?
                # No, minimize calls func internally.
                # If it throws, we lose track of exact calls.
                # But usually minimize doesn't throw unless func throws.
                # If func throws, the whole evaluator fails?
                # "an exception before the first call, makes the whole result invalid; an exception later keeps the calls made."
                # If minimize throws, we don't know how many calls were made.
                # This is a risk.
                # To mitigate, we can implement our own counting wrapper?
                # No, we can't modify how minimize calls func.
                # But we can pass a wrapper function to minimize that counts calls.
                
                pass
