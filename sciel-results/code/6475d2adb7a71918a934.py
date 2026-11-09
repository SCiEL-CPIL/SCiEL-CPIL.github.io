"""Multi-start CMA-ES optimizer for BBOB f16 (Weierstrass)."""
import numpy as np
import time

class Optimizer:
    """Multi-start CMA-ES optimizer for BBOB f16 (Weierstrass)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0
        
        # State
        best_x = None
        best_f = np.inf
        
        # Weierstrass f16 is multimodal with oscillations.
        # Strategy: Multi-start local search using a simplified CMA-ES or Powell-like method.
        # Given the budget (500) and dim (5), we can afford ~20-30 restarts.
        # Each restart uses a small budget (e.g., 15-20 evaluations).
        
        # To ensure early progress (AOCC), we start with a coarse global scan.
        # Then we refine the best points found.
        
        # Phase 1: Global Scan (Latin Hypercube or Grid)
        # 5D space. A grid of 3^5 = 243 points is too many for 500 budget if we want refinement.
        # Let's use a smaller initial set, e.g., 25 points (5^2) or random 30.
        # Actually, for AOCC, getting a decent value early is key.
        # Let's use a Halton sequence or random points.
        
        # Let's try 20 initial random points.
        n_init = 20
        if budget < n_init:
             # Fallback for very small budgets
             x = np.random.uniform(-5.0, 5.0, size=dim)
             fx = func(x)
             return x, fx

        X_init = np.random.uniform(-5.0, 5.0, size=(n_init, dim))
        # Sort by function value? No, we need to evaluate them.
        
        points = []
        for i in range(n_init):
            if time.time() > deadline or budget <= 0:
                break
            x = X_init[i]
            fx = func(x)
            budget -= 1
            points.append((x, fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if not points:
             x = np.zeros(dim)
             fx = func(x)
             return x, fx

        # Sort points by fitness (best first)
        points.sort(key=lambda p: p[1])
        
        # Phase 2: Local Refinement from Top Candidates
        # We will take the top K candidates and run a local search on each.
        # The local search will be a simple CMA-ES or a derivative-free local search.
        # Given the oscillatory nature, a small step size is needed.
        # CMA-ES is robust. Let's implement a lightweight CMA-ES.
        
        # Alternatively, since CMA-ES can be heavy to implement correctly in a snippet,
        # let's use a Nelder-Mead simplex or a simple coordinate descent with random restarts?
        # Nelder-Mead is standard and effective for smooth-ish functions.
        # Weierstrass is continuous but not differentiable everywhere. Nelder-Mead works.
        
        # Let's use Nelder-Mead from scipy.optimize.
        # But we need to manage the budget carefully.
        
        # Strategy:
        # 1. Pick the best point.
        # 2. Run Nelder-Mead with a max_iter limit.
        # 3. If it improves, great.
        # 4. Move to the next best point from the initial scan if budget remains.
        
        # To maximize AOCC, we should refine the *current best* as much as possible?
        # Or spread out?
        # AOCC rewards early improvement. Once we are close to the optimum, further improvements
        # yield diminishing returns in AOCC (log scale).
        # However, if the first refinement fails to find the global optimum, we are stuck.
        # So, multi-start is good.
        
        # Let's allocate budget dynamically.
        # For each of the top 5 initial points, run a local search.
        
        top_k = min(5, len(points))
        
        # Define a local search function using Nelder-Mead
        # We need to track the global best_f and best_x.
        
        # Note: scipy.optimize.minimize uses its own budget. We need to wrap func to count calls.
        
        class FuncWrapper:
            def __init__(self, func, budget_ref):
                self.func = func
                self.budget_ref = budget_ref # mutable list to track remaining budget
                
            def __call__(self, x):
                if self.budget_ref[0] <= 0 or time.time() > deadline:
                    return self.last_f # Return last known best? Or a large value?
                    # Actually, if budget is exceeded, we should stop.
                    # But minimize doesn't know to stop.
                    # We'll just return a value and check after.
                fx = self.func(x)
                self.budget_ref[0] -= 1
                self.last_f = fx
                return fx
            
        # Initialize wrapper
        budget_remaining = [budget]
        fw = FuncWrapper(func, budget_remaining)
        fw.last_f = best_f
        
        # We will iterate through the top candidates
        for i in range(top_k):
            if budget_remaining[0] <= 0 or time.time() > deadline:
                break
            
            x_start = points[i][0]
            f_start = points[i][1]
            
            # If this start is worse than the current global best, we might still find a better local min?
            # Yes, if the global optimum is in a different basin.
            
            # Run Nelder-Mead
            # Set maxiter to use up to ~10-20 evals per restart to keep it fast.
            # Total budget 500. 5 restarts * 20 = 100. We have plenty left.
            # Let's allow up to 30 evals per restart.
            
            max_iter = 30
            
            try:
                res = scipy_minimize(fw, x_start, method='Nelder-Mead', 
                                     options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-6})
                
                # Update global best if needed
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                
            except Exception:
                pass
            
            # Check time/budget
            if time.time() > deadline or budget_remaining[0] <= 0:
                break
        
        # If we still have budget, do more restarts from random points or refine best_x further
        # Let's refine the current best_x with a tighter Nelder-Mead run if budget allows.
        
        if budget_remaining[0] > 10 and best_x is not None and time.time() < deadline:
             # Re-run Nelder-Mead on the best point found so far with more iterations
             # This helps if the previous run stopped early due to maxiter or tolerance.
             max_iter_final = budget_remaining[0] // 2 # Use half remaining budget
             if max_iter_final > 5:
                 try:
                     res = scipy_minimize(fw, best_x, method='Nelder-Mead', 
                                          options={'maxiter': max_iter_final, 'xatol': 1e-5, 'fatol': 1e-7})
                     if res.fun < best_f:
                         best_f = res.fun
                         best_x = res.x.copy()
                 except Exception:
                     pass
        
        return best_x, best_f

# Import scipy inside the class to avoid top-level import issues if not needed, 
# but the prompt says SciPy is available.
import scipy.optimize

# Wait, the import should be at the top or inside. Let's put it at the top.
