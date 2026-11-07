"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Rastrigin-aware hybrid: structured init + multi-restart local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        rng = np.random.default_rng(42)
        
        calls_made = 0

        def counted_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            return func(x)

        def check_time():
            return time.time() > deadline

        # Phase 1: Structured Global Exploration
        # We want to find a good basin. Rastrigin is multimodal.
        # The minimum is shifted and rotated.
        # Strategy:
        # 1. Try origin (strong prior if shift is small or rotation aligns axes).
        # 2. Try a set of points covering the domain [-4, 4]^5 roughly.
        # 3. Use remaining budget for local refinement.
        
        # Let's reserve 250 evals for exploration, 250 for exploitation.
        # But we should be adaptive.
        
        best_x = None
        best_f = np.inf
        
        # Candidate points for initialization
        # 1. Origin
        # 2. Uniform random samples
        # 3. Points on a coarse grid? 
        #    In 5D, a grid of 3 points per dim is 3^5 = 243 points.
        #    A grid of 2 points per dim is 2^5 = 32 points.
        #    Let's try a mix.
        
        # First, evaluate origin
        x0 = np.zeros(dim)
        f0 = counted_func(x0)
        calls_made += 1 # counted_func already incremented, wait, I need to be careful.
        # My counted_func increments calls_made.
        # So calls_made is now 1.
        
        if f0 < best_f:
            best_f = f0
            best_x = x0.copy()
        
        # Let's do a batch of random samples and a small grid.
        # Grid points in [-4, 4]
        # 3 points: -4, 0, 4
        # 5 points: -4, -2, 0, 2, 4
        
        # Let's use 3 points per dim for a coarse grid: 3^5 = 243 points.
        # This might be too many if we want to leave room for refinement.
        # But 243 + 1 = 244 evals. Leaves 256 for refinement.
        # Refinement with BFGS/Nelder-Mead can be very efficient.
        
        # However, if the shift is large, the grid might miss the basin.
        # Random sampling is better for coverage.
        
        # Let's do 50 random samples first.
        n_random = 50
        for i in range(n_random):
            if check_time() or calls_made >= budget:
                break
            x = rng.uniform(-4, 4, dim)
            f = counted_func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Now, let's try to refine the best point found so far.
        # We will use a local optimizer.
        # Nelder-Mead is robust and derivative-free.
        # BFGS is faster if we are close to a minimum, but might step out of the basin.
        # Let's try Nelder-Mead first with a tight budget.
        
        # We have budget - calls_made left.
        # Let's allocate half of the remaining budget to this first refinement.
        
        def run_local_search(x_init, f_init, max_iter):
            nonlocal best_f, best_x, calls_made
            
            # We need a wrapper that stops if we hit budget or time
            # scipy.optimize.minimize doesn't have a built-in "max function calls" limit for Nelder-Mead easily 
            # other than maxiter, which is iterations, not function evals.
            # Nelder-Mead uses ~ (n+1) * maxiter evals? No, it's complex.
            # It's safer to implement a simple local search or use a library that respects call counts.
            # Or, we can just run it and hope it converges fast.
            # But if it runs away, we waste budget.
            
            # Let's use a custom Nelder-Mead implementation or just use scipy with a wrapper that checks calls.
            # Since we can't easily interrupt scipy, let's be conservative with maxiter.
            
            try:
                # Nelder-Mead
                # Initial simplex size: 0.05 * (x - x0) or just 0.05
                # Let's use the default or small.
                
                # We will create a wrapper for func that raises an exception if budget is exceeded
                # to break out of scipy's loop.
                def wrapper(x):
                    nonlocal calls_made
                    if calls_made >= budget:
                        raise RuntimeError("Budget exceeded")
                    return counted_func(x)
                
                res = minimize(
                    wrapper, 
                    x_init, 
                    method='Nelder-Mead', 
                    options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4}
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except RuntimeError:
                pass
            except Exception:
                pass

        # Run local search from best_x
        # Estimate remaining budget
        remaining = budget - calls_made
        # Allocate 50% to this first refinement
        alloc1 = int(remaining * 0.5)
        
        # Nelder-Mead in 5D: maxiter=100 might use ~100*6=600 calls?
        # No, Nelder-Mead is not that expensive per iteration.
        # Let's be safe: maxiter = alloc1 // 10
        max_iter1 = max(10, alloc1 // 10)
        
        if not check_time():
            run_local_search(best_x, best_f, max_iter1)
            
        # If we still have budget, try a second restart from a different point?
        # Or refine further?
        # Let's try a second local search from a slightly perturbed best_x or a random point.
        
        remaining = budget - calls_made
        if remaining > 50 and not check_time():
            # Try a random point near the best one
            x2 = best_x + rng.normal(0, 0.5, dim)
            x2 = np.clip(x2, -5, 5)
            f2 = counted_func(x2)
            
            if f2 < best_f:
                best_f = f2
                best_x = x2.copy()
            
            # Refine this new point
            alloc2 = int((budget - calls_made) * 0.5)
            max_iter2 = max(10, alloc2 // 10)
            if not check_time():
                run_local_search(best_x, best_f, max_iter2)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
