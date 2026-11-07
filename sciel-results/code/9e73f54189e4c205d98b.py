"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """(1+1)-ES with multi-start initialization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Safety margin for time
        start_time = time.time()
        deadline = start_time + 290.0
        
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        
        # Wrapper to count calls and handle time
        def wrapped_func(x):
            nonlocal calls_made, best_f, best_x
            if time.time() > deadline:
                return best_f
            if calls_made >= budget:
                return best_f
            
            x = np.clip(x, -5, 5)
            f_val = func(x)
            calls_made += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
                
            return f_val

        # Phase 1: Multi-start Initialization
        # Weierstrass f16 is multimodal but the hidden shift/rotation keeps the minimizer in [-4, 4]^5.
        # We perform a coarse grid search to find a good basin.
        
        # Strategy: 
        # 1. Evaluate center (0,0,0,0,0)
        # 2. Evaluate a few random points
        # 3. For the best few points, run Nelder-Mead to refine.
        
        rng = np.random.default_rng(42)
        
        # Initial candidates
        candidates = [np.zeros(dim)]
        # Add some random points
        for _ in range(10):
            candidates.append(rng.uniform(-4, 4, dim))
            
        # Evaluate initial candidates
        for x_cand in candidates:
            if time.time() > deadline or calls_made >= budget:
                break
            wrapped_func(x_cand)
            
        # If we didn't even find the center, fallback
        if best_f == np.inf:
             best_f = wrapped_func(np.zeros(dim))

        # Sort candidates by function value? We don't store them separately easily without extra memory.
        # Instead, we will run Nelder-Mead from the current best, and then maybe from a few other good ones if we kept track.
        # To keep it simple and robust:
        # Run Nelder-Mead from the current best point.
        
        # Nelder-Mead parameters
        # Weierstrass is smooth, so NM should work well if started in the basin.
        # If started outside, it might struggle, hence the multi-start.
        
        # Let's do 3 restarts of NM if budget allows.
        # Restart 1: From current best
        # Restart 2: From a random point near best
        # Restart 3: From a random point
        
        # We need to be careful with the budget.
        # NM can take many function evaluations.
        # We will use a custom callback or just let it run and check time.
        # SciPy's minimize doesn't have a strict "max time" stop, only max iterations.
        # We'll set maxiter based on remaining budget roughly, but the wrapped_func will stop it.
        
        # Let's define a helper to run NM from a point
        def run_nm(x0):
            nonlocal calls_made, best_f, best_x
            if time.time() > deadline or calls_made >= budget:
                return
            
            # NM options
            # Weierstrass has small features. 
            # Initial simplex size: 0.05 + 0.05 * abs(x) is standard, but we can try 0.1
            try:
                res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                               options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 1000})
            except Exception:
                pass

        # 1. Refine current best
        run_nm(best_x)
        
        # 2. Try a second start: slightly perturbed best
        if time.time() < deadline and calls_made < budget:
            x2 = best_x + 0.5 * rng.standard_normal(dim)
            x2 = np.clip(x2, -5, 5)
            # Quick check if x2 is better than current best before running NM?
            # If we just run NM, it will evaluate x2 as part of the initial simplex.
            # But if x2 is terrible, NM might waste time.
            # Let's evaluate x2 first.
            f2 = wrapped_func(x2)
            if f2 < best_f:
                run_nm(x2)
            else:
                # If x2 is worse, maybe try a different direction?
                # Or just try a random point in the box.
                pass

        # 3. Try a third start: random point in [-4, 4]
        if time.time() < deadline and calls_made < budget:
            x3 = rng.uniform(-4, 4, dim)
            f3 = wrapped_func(x3)
            if f3 < best_f:
                run_nm(x3)
            else:
                # If this random point is better than the current best, run NM from it.
                # If not, we might have found a local min. 
                # Weierstrass is multimodal. 
                # Let's try one more random point if we have budget.
                pass
                
        # 4. If we have lots of budget left, do a final polish with (1+1)-ES from best
        if time.time() < deadline and calls_made < budget:
             x = best_x
             fx = best_f
             sigma = 0.1
             # Simple (1+1)-ES for final polish
             while calls_made < budget and time.time() < deadline:
                 y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                 fy = wrapped_func(y)
                 if fy < fx:
                     x, fx = y, fy
                     sigma *= 1.1
                 else:
                     sigma *= 0.9
                 if sigma < 1e-6:
                     break
             # Update global best if ES improved it
             if fx < best_f:
                 best_f = fx
                 best_x = x

        return best_x, best_f
# EVOLVE-BLOCK-END
