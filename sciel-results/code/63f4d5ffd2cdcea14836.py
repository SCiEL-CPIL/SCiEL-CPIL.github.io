import numpy as np
import scipy.optimize as opt
import time

class Optimizer:
    """Multi-start Nelder-Mead optimizer for Katsuura function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s limit, stop at 290s
        deadline = start_time + 290.0
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        used_calls = 0
        max_calls = self.budget
        
        # Helper to track calls and enforce budget/time
        def safe_call(x):
            nonlocal used_calls, best_f, best_x
            if used_calls >= max_calls:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("Time limit exceeded")
            
            # Clip to be safe, though func does it
            x_clipped = np.clip(x, -5, 5)
            used_calls += 1
            f_val = func(x_clipped)
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
                
            return f_val

        # Phase 1: Global Exploration
        # Sample a diverse set of points to find good starting candidates
        # Katsuura is multimodal, so we need a good spread.
        n_initial = 20
        # Ensure we don't exceed budget in initial phase
        n_initial = min(n_initial, max_calls // 2)
        
        initial_points = rng.uniform(-5, 5, size=(n_initial, self.dim))
        initial_values = []
        
        for p in initial_points:
            if used_calls >= max_calls or time.time() > deadline:
                break
            try:
                fp = safe_call(p)
                initial_values.append((fp, p))
            except RuntimeError:
                break
        
        if best_x is None:
            return None, float('inf')

        # Phase 2: Multi-Start Local Optimization
        # Sort initial points by function value
        initial_values.sort(key=lambda x: x[0])
        
        # We will run Nelder-Mead from the top 'k' best initial points
        # k depends on remaining budget
        remaining_budget = max_calls - used_calls
        
        if remaining_budget > 10:
            # Determine number of restarts
            # Each NM run needs at least dim+1 calls to form simplex, plus iterations
            # Let's aim for at most 5 restarts, or as many as budget allows
            # A typical NM run in 5D might take 50-100 calls to converge reasonably
            max_restarts = 5
            budget_per_restart = remaining_budget // max_restarts
            
            # If budget per restart is too small, reduce restarts
            if budget_per_restart < 20:
                max_restarts = max(1, remaining_budget // 20)
                budget_per_restart = remaining_budget // max_restarts
            
            # Select top candidates from initial_values
            # If we have fewer initial points than max_restarts, use all
            candidates = initial_values[:max_restarts]
            
            # If we have more restarts allowed than candidates, add some random points
            # to ensure diversity if initial exploration was poor
            while len(candidates) < max_restarts and used_calls + 1 < max_calls:
                x_rand = rng.uniform(-5, 5, self.dim)
                try:
                    fp_rand = safe_call(x_rand)
                    candidates.append((fp_rand, x_rand))
                except RuntimeError:
                    break
            
            for fp, x_start in candidates:
                if used_calls >= max_calls or time.time() > deadline:
                    break
                
                local_budget = max_calls - used_calls
                if local_budget < 10:
                    break
                
                try:
                    # Use Nelder-Mead
                    # maxfev is the maximum number of function calls
                    # maxiter is the maximum number of iterations
                    # We set maxfev to local_budget to strictly respect the limit
                    result = opt.minimize(
                        safe_call, 
                        x_start, 
                        method='Nelder-Mead',
                        options={
                            'maxiter': local_budget, 
                            'maxfev': local_budget,
                            'xatol': 1e-8,
                            'fatol': 1e-8,
                            'adaptive': True
                        }
                    )
                    
                    # Note: safe_call updates best_f/best_x internally
                    # result.fun is the last value, but best_f tracks the minimum
                    # so we don't need to explicitly update here unless safe_call failed to track
                    # (which it doesn't, it tracks globally)
                    
                except (RuntimeError, Exception):
                    # If budget or time exceeded, stop further restarts
                    # The best_x/best_f found so far is preserved
                    if time.time() > deadline or used_calls >= max_calls:
                        break
                    # If it was just a convergence failure, continue to next restart
                    continue
        
        # Phase 3: Final Polish (if budget remains)
        # If we still have significant budget left, do a final NM run from the global best
        if used_calls < max_calls and time.time() < deadline:
            final_budget = max_calls - used_calls
            if final_budget > 10:
                try:
                    result = opt.minimize(
                        safe_call, 
                        best_x, 
                        method='Nelder-Mead',
                        options={
                            'maxiter': final_budget,
                            'maxfev': final_budget,
                            'xatol': 1e-10,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                except (RuntimeError, Exception):
                    pass

        return best_x, best_f
