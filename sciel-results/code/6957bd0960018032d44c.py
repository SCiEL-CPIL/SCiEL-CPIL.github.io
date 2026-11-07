import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 290.0
        
        budget = self.budget
        dim = self.dim
        calls = 0
        best_f = float('inf')
        best_x = None
        
        # Helper to evaluate and track
        def evaluate(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            try:
                val = func(x)
            except Exception:
                return None
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val
        
        # Phase 1: Global exploration
        # Use a moderate number of points to identify promising basins.
        # 25 points is a good balance for dim=5, budget=500.
        n_explore = 25
        candidates = []
        
        for _ in range(n_explore):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = evaluate(x)
            if fx is not None:
                candidates.append((fx, x))
        
        if best_x is None:
            return None, None
        
        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Multi-start Nelder-Mead
        # Take the top 5 starting points.
        top_k = min(5, len(candidates))
        
        # Calculate remaining budget
        remaining_budget = budget - calls
        
        if remaining_budget <= 0:
            return best_x, best_f
        
        # Allocate budget to each start
        # We leave a small buffer for safety
        budget_per_run = max(1, (remaining_budget - 10) // top_k)
        
        for i in range(top_k):
            if calls >= budget or time.time() > deadline:
                break
            
            x0 = candidates[i][1]
            f0 = candidates[i][0]
            
            # Nelder-Mead options
            # maxiter is an upper bound on function evaluations roughly.
            # We set it slightly higher than budget_per_run to allow the optimizer
            # to stop naturally if it converges, but the evaluate() function
            # will cut it off if the global budget is hit.
            opts = {
                'maxiter': budget_per_run + 10,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True
            }
            
            try:
                res = minimize(func, x0, method='Nelder-Mead', options=opts)
                # Check if res.fun is valid and better
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
