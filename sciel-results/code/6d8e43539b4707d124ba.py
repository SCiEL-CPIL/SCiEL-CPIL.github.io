import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton_sequence(self, n, dim):
        """Generate Halton sequence points in [0, 1]^dim."""
        points = []
        for i in range(1, n + 1):
            point = []
            for base in range(2, 2 + dim):
                f = 1.0
                r = 0.0
                i_temp = i
                while i_temp > 0:
                    f /= base
                    r += f * (i_temp % base)
                    i_temp //= base
                point.append(r)
            points.append(point)
        return np.array(points)

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin
        d = self.dim
        total_budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        history = [] # List of (f, x)

        def track_call(x):
            nonlocal best_x, best_f, calls_used
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            calls_used += 1
            history.append((f, x_clipped.copy()))
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # Phase 1: Halton Initialization
        # Use 50% of the budget for exploration? No, Halton is cheap. 
        # Let's use 50 points for initial sampling.
        n_halton = min(50, total_budget)
        halton_pts = self._halton_sequence(n_halton, d)
        for pt in halton_pts:
            if calls_used >= total_budget or time.time() > deadline:
                break
            # Scale to [-5, 5]
            x = -5.0 + 10.0 * pt
            track_call(x)

        if best_x is None:
            track_call(np.zeros(d))

        # Phase 2: Local Optimization from Top Candidates
        # Sort history by function value
        history.sort(key=lambda item: item[0])
        
        # We will try to optimize from the top K candidates.
        # Budget remaining: total_budget - calls_used
        # Each Nelder-Mead run might use 50-100 calls.
        # Let's allocate budget dynamically.
        
        remaining_budget = total_budget - calls_used
        if remaining_budget <= 0:
            return best_x, best_f

        # Select top 5 unique candidates (avoid duplicates if any, though Halton is unique)
        top_candidates = [x for f, x in history[:5]]
        
        # Allocate calls to each candidate
        # If we have 450 calls left, and 5 candidates, we can give 90 to each.
        # But we should prioritize the best one.
        # Let's give 50% of remaining to the best, and distribute the rest.
        
        # Actually, a simpler robust strategy:
        # Run Nelder-Mead from the best point with a generous budget.
        # If it finishes early, run from the next best.
        
        # Define a wrapper that stops when budget is hit or time is up
        local_calls = [0]
        
        def local_func(x):
            if local_calls[0] >= remaining_budget or time.time() > deadline:
                return np.inf
            local_calls[0] += 1
            return func(np.clip(x, -5.0, 5.0))

        for idx, start_x in enumerate(top_candidates):
            if calls_used + local_calls[0] >= total_budget:
                break
            if time.time() > deadline:
                break
            
            # Reset local call counter for this run? No, we need to track global calls.
            # The wrapper above increments local_calls. We need to map this to global calls.
            # Better: use a closure that checks global calls_used.
            
            def wrapped_func(x):
                nonlocal calls_used
                if calls_used >= total_budget or time.time() > deadline:
                    return np.inf
                calls_used += 1
                return func(np.clip(x, -5.0, 5.0))

            # Nelder-Mead options
            # maxiter controls the number of iterations, not exactly calls, but roughly 2*d per iter?
            # In 5D, 100 iterations is ~500 calls? No, Nelder-Mead is very efficient.
            # Let's set maxiter to be large and rely on the call limit in the wrapper.
            # xatol and fatol for convergence.
            
            try:
                res = minimize(
                    wrapped_func, 
                    start_x, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200, 
                        'xatol': 1e-8, 
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                # Update best if found
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

            # If we still have budget and time, try the next candidate
            # But if the first one used up the budget, the loop will break.

        return best_x, best_f
