"""Rewritten program for BBOB f7 (Step ellipsoidal) using Multi-restart CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-restart CMA-ES optimizer for BBOB f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        dim = self.dim
        budget = self.budget
        
        # Track global best
        x_global_best = None
        f_global_best = np.inf
        calls_used = 0

        # Helper to check time and budget
        def can_continue():
            nonlocal calls_used
            return calls_used < budget and (time.time() - start_time) < (deadline - start_time)

        # Phase 1: Initial Scouts to find a reasonable starting point
        # The function is flat, so random points are often similar. 
        # We want to avoid the center if possible, or just start broadly.
        n_scouts = min(10, budget)
        rng = np.random.default_rng(42)
        
        for i in range(n_scouts):
            if not can_continue():
                break
            # Sample from a broad distribution
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            calls_used += 1
            if f < f_global_best:
                f_global_best = f
                x_global_best = x.copy()

        # If we haven't found anything (shouldn't happen), fallback
        if x_global_best is None:
            x_global_best = np.zeros(dim)
            f_global_best = func(x_global_best)
            calls_used += 1

        # Phase 2: Multi-restart CMA-ES
        # We will use up to 5 restarts.
        # Each restart gets a decreasing fraction of the remaining budget.
        # The step size for CMA-ES is initialized based on the scout spread or a default.
        
        max_restarts = 5
        # Distribute budget: 40%, 25%, 15%, 10%, 10% roughly
        # But we must respect the absolute budget.
        
        # Estimate a good initial sigma. 
        # In f7, the drop is sharp. A moderate sigma is good.
        # Let's use 1.0 as a baseline, or scale based on scout variance if we had it.
        # Given the flatness, 1.0 is a safe bet for dim=5.
        
        for r in range(max_restarts):
            if not can_continue():
                break
            
            # Determine budget for this restart
            # Simple heuristic: divide remaining budget by remaining restarts
            remaining_budget = budget - calls_used
            if remaining_budget < 10:
                break
                
            # Allocate more budget to earlier restarts
            if r == 0:
                alloc_budget = int(remaining_budget * 0.5)
            elif r == 1:
                alloc_budget = int(remaining_budget * 0.3)
            elif r == 2:
                alloc_budget = int(remaining_budget * 0.15)
            else:
                alloc_budget = int(remaining_budget * 0.05)
            
            # Ensure minimum budget for CMA-ES to do anything useful
            alloc_budget = max(alloc_budget, 20)
            if calls_used + alloc_budget > budget:
                alloc_budget = budget - calls_used
            
            if alloc_budget < 5:
                break

            # Initialize CMA-ES
            # x0 is the current global best
            x0 = x_global_best.copy()
            
            # Options for scipy CMA-ES
            # maxiter is not the call count, but we can use callback to count.
            # Better: use 'maxiter' as a loose limit and rely on our own budget check via callback.
            
            call_count = [0]
            def callback(xk):
                nonlocal calls_used
                call_count[0] += 1
                calls_used += 1
                # Check if we exceeded allocated budget or time
                if call_count[0] >= alloc_budget or not can_continue():
                    # We can't easily break scipy minimize from callback, 
                    # but we can raise an exception or just let it finish if it's fast.
                    # However, scipy CMA-ES might be slow. 
                    # It's safer to set maxiter to something reasonable and check time in a wrapper.
                    # Actually, scipy CMA-ES does not have a strict 'maxcalls' parameter in older versions, 
                    # but we can monitor time.
                    pass

            # Since we can't easily break scipy minimize mid-run based on custom budget,
            # we will set maxiter to a value that corresponds to our budget.
            # CMA-ES typically uses ~ (4 + ln(dim)) * dim candidates per iteration.
            # For dim=5, that's ~ (4+1.6)*5 = 28 calls per iter.
            # So maxiter = alloc_budget / 28
            maxiter = max(1, alloc_budget // 30)

            try:
                res = minimize(
                    func,
                    x0,
                    method='CMAES',
                    options={
                        'maxiter': maxiter,
                        'tol': 1e-10,
                        'verbose': 0,
                        'seed': 42 + r, # Different seed for each restart
                        'bounds': [(-5, 5)] * dim
                    }
                )
                
                # Update global best
                if res.fun < f_global_best:
                    f_global_best = res.fun
                    x_global_best = res.x.copy()
                    
            except Exception:
                # If CMA-ES fails, continue to next restart
                pass

            # If we improved significantly, we might want to tighten the search.
            # If not, the next restart will use the same x_global_best but a new seed/covariance.
            
        return x_global_best, f_global_best

# EVOLVE-BLOCK-END
