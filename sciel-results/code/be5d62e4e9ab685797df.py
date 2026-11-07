import numpy as np
import time
import itertools

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        lower, upper = -5.0, 5.0
        
        start_time = time.time()
        deadline = start_time + 295.0  # 35s safety margin before 330s kill

        calls_made = 0
        best_x = None
        best_f = float('inf')
        all_results = []  # list of (f, x)

        def track(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                raise ValueError("Budget exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, lower, upper)
            
            try:
                f = func(x)
            except Exception as e:
                raise TimeoutError("Internal error or budget") from e
            
            calls_made += 1
            all_results.append((f, x.copy()))
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
            
            return f

        # Phase 1: Structured Coarse Search
        # Minimizer is in [-4, 4]^5.
        # We use a 3-level grid: -4, 0, 4.
        # 3^5 = 243 points.
        # Sort by distance to origin to maximize early AOCC (center is likely good).
        levels = [-4.0, 0.0, 4.0]
        grid_points = list(itertools.product(levels, repeat=dim))
        origin = np.zeros(dim)
        grid_points.sort(key=lambda p: np.linalg.norm(np.array(p) - origin))

        for p in grid_points:
            if calls_made >= budget or time.time() > deadline:
                break
            try:
                track(np.array(p, dtype=np.float64))
            except (TimeoutError, ValueError):
                break
            except Exception:
                break

        # Phase 2: Multi-scale L-BFGS-B Refinement
        # Strategy: Use scipy.optimize.minimize with L-BFGS-B.
        # L-BFGS-B is fast and robust for smooth BBOB functions.
        # We perform multi-scale restarts: start with large steps (via bounds or scaling) 
        # and refine. Since L-BFGS-B works on the original scale, we can just run it 
        # from multiple starting points found in Phase 1.
        
        # We will try to refine the top K candidates from the grid search.
        # K is chosen to balance exploration of different basins vs depth of refinement.
        
        # Sort all results by function value
        all_results.sort(key=lambda t: t[0])
        
        # We have used ~243 calls. Remaining: ~257 calls.
        # Let's allocate budget to multiple local optimizations.
        # L-BFGS-B typically converges in 20-50 evals.
        # Let's try refining the top 5 candidates.
        
        top_k = 5
        candidates = [r[1] for r in all_results[:top_k]]
        
        # Ensure we don't refine if we're out of budget
        if calls_made >= budget:
            return best_x, best_f

        # For each candidate, run L-BFGS-B with a limited budget
        # We need to be careful with the total budget.
        # Total remaining: budget - calls_made
        # Let's divide this among the candidates, but give more to the best ones.
        
        remaining = budget - calls_made
        if remaining <= 0:
            return best_x, best_f
        
        # Allocate evals: 40% to best, 30% to second, 30% to others? 
        # Or just a fixed amount per candidate, e.g., 50 evals each?
        # 5 candidates * 50 evals = 250 evals. Fits in 257.
        
        max_evals_per_candidate = 50
        
        for idx, x0 in enumerate(candidates):
            if calls_made >= budget or time.time() > deadline:
                break
            
            # If this candidate is already worse than the current best by a lot, 
            # and it's not the first candidate, we might skip it to save budget for the best one?
            # But for multimodal, it might be in a different basin. 
            # Let's stick to the plan: refine top 5.
            
            # Skip if x0 is the same as the previous x0 (unlikely with grid)
            
            # Run L-BFGS-B
            # We use a wrapper to track calls and respect the global budget/deadline.
            # scipy.optimize.minimize doesn't have a strict "max_func_evals" that stops 
            # immediately on exception in a way that plays well with our global counter 
            # unless we use a custom callback or wrap the function.
            # Wrapping the function is safer.
            
            local_calls = 0
            max_local_calls = max_evals_per_candidate
            
            def local_func(x):
                nonlocal local_calls
                if local_calls >= max_local_calls:
                    # Return a large value to discourage further exploration?
                    # Or raise an exception? 
                    # Raising an exception in scipy might be messy. 
                    # Better to just return the current best or a high value?
                    # Actually, if we exceed local budget, we should stop the optimization.
                    # We can raise a specific exception and catch it.
                    raise TimeoutError("Local budget exceeded")
                
                local_calls += 1
                # Check global constraints
                if calls_made >= budget or time.time() > deadline:
                    raise TimeoutError("Global budget or time exceeded")
                
                return track(x)
            
            try:
                from scipy.optimize import minimize
                res = minimize(
                    local_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=[(lower, upper)] * dim,
                    options={'maxiter': 100, 'maxfun': max_local_calls + 10} # Slight buffer
                )
                # The result of minimize is ignored, we rely on track() to update best_x/best_f
            except (TimeoutError, ValueError):
                pass
            except Exception:
                # Catch any other scipy errors
                pass

        return best_x, best_f
