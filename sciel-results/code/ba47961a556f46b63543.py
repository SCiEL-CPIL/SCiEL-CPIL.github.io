import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        bounds = [(-5, 5)] * dim
        
        # State tracking
        best_x = None
        best_f = float('inf')
        call_count = 0
        start_time = time.time()
        
        # Safety margin: stop if we are close to the 300s limit or budget limit
        # The task says 300s limit, but usually these tasks are fast. 
        # The 500 call limit is the hard constraint for validity.
        
        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            call_count += 1
            # Check time limit loosely
            if time.time() - start_time > 280:
                # We can't easily break out of scipy minimize from inside the callback
                # But we can return a large value to discourage further steps? 
                # Actually, better to just let it run if it's fast. 
                # 500 calls is very fast for L-BFGS-B in 5D.
                pass
            
            # Clip input to bounds to be safe, though L-BFGS-B should respect them
            x_clipped = np.clip(x, -5, 5)
            f_val = float(func(x_clipped))
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        # Generate starting points using Halton sequence
        # Number of restarts
        # We need to balance exploration (more restarts) vs exploitation (more evals per restart)
        # 500 evals total.
        # L-BFGS-B in 5D might converge in ~10-30 evals if close, or more if far.
        # Let's try 10 restarts.
        n_restarts = 10
        # Reserve some budget for final polish
        polish_budget = 50
        search_budget = total_budget - polish_budget
        if search_budget < n_restarts:
            # Fallback if budget is tiny
            n_restarts = max(1, search_budget // 5)
            polish_budget = 0
            
        evals_per_restart = max(5, search_budget // n_restarts)
        
        # Halton sequence generator
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(n, dim):
            points = []
            for i in range(1, n + 1):
                point = []
                # Use first few primes for bases
                bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
                for d in range(dim):
                    point.append(halton(i, bases[d]))
                points.append(point)
            return np.array(points)

        # Generate points in [0, 1]^dim
        halton_points = halton_sequence(n_restarts, dim)
        # Map to [-5, 5]^dim
        start_points = halton_points * 10.0 - 5.0

        # Run L-BFGS-B from each start point
        for i in range(n_restarts):
            if call_count >= total_budget:
                break
                
            x0 = start_points[i]
            
            # Ensure x0 is within bounds
            x0 = np.clip(x0, -5, 5)
            
            # Determine maxfev for this run
            # We don't want to exceed total_budget
            remaining = total_budget - call_count
            # Leave a little bit for the next restarts? 
            # Simple approach: limit this run to evals_per_restart or remaining, whichever is smaller
            # But we also want to save polish_budget.
            # So limit to min(evals_per_restart, remaining - (n_restarts - i - 1) * 5) ? 
            # Too complex. Just use evals_per_restart.
            # If it uses all of it, we might run out of budget for later restarts.
            # Better: distribute remaining search budget evenly among remaining restarts.
            
            restarts_left = n_restarts - i
            available_for_search = total_budget - call_count - polish_budget
            if available_for_search < 0:
                 available_for_search = total_budget - call_count
                 
            if restarts_left > 0:
                current_maxfev = max(1, available_for_search // restarts_left)
            else:
                current_maxfev = 1
                
            try:
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 100, 'maxfev': current_maxfev, 'ftol': 1e-10, 'gtol': 1e-8}
                )
            except Exception:
                # If minimize fails, continue
                continue
            
            # Update best if this run found something better (wrapped_func already does this, 
            # but res.fun might be better if the last step was good)
            if res.fun < best_f:
                best_f = res.fun
                best_x = np.clip(res.x, -5, 5)

        # Final Polish
        if best_x is not None and call_count < total_budget - 1:
            remaining = total_budget - call_count
            if remaining > 0:
                try:
                    res = minimize(
                        wrapped_func, 
                        best_x, 
                        method='L-BFGS-B', 
                        bounds=bounds,
                        options={'maxiter': 100, 'maxfev': remaining, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        # If no best_x found (shouldn't happen), return a default
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x) # This consumes 1 call, but we are out of budget? 
            # If we are out of budget, we shouldn't call. 
            # But if call_count == 0, we haven't called anything.
            # The loop above should have called at least once.
            # If n_restarts was 0, we skip.
            # If budget is 0, we can't call.
            # Let's assume budget >= 1.

        return best_x, best_f
