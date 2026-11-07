import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim
        
        # Use a fixed seed for reproducibility across instances if needed, 
        # but the task implies different hidden shifts per instance.
        # However, deterministic search patterns are often robust.
        # Let's use a seed based on nothing specific, or just 0.
        rng = np.random.default_rng(0)
        
        best_x = None
        best_f = np.inf
        eval_count = 0
        
        # Strategy: Multi-Start CMA-ES
        # Katsuura is multimodal. DE with NP=20 might be too slow to converge 
        # or might get stuck. CMA-ES is more sample-efficient for continuous 
        # non-linear optimization, especially when the landscape has some 
        # structure (even if multimodal).
        
        # We split the budget into several "starts".
        # Each start initializes CMA-ES at a different location.
        # We use a Sobol-like or grid-like distribution for initial points to ensure coverage.
        
        # For dim=5, a 3x3x3x3x3 grid is 243 points, which is too many for 500 budget 
        # if we want to run CMA-ES properly (needs ~100-200 evals per run to converge).
        # Let's do 5-10 starts.
        
        num_starts = 8
        # Reserve some budget for a final polishing or just let the last start run longer?
        # Better: Fixed budget per start, but if time allows, run more?
        # No, strict budget. 
        # Budget per start = budget // num_starts
        # But CMA-ES overhead is high.
        # Let's allocate:
        # Start 1: 50 evals
        # Start 2: 50 evals
        # ...
        # Start 8: 50 evals
        # Total 400. Remaining 100 for a final global refinement or just wasted?
        # Actually, CMA-ES can be stopped early if convergence is reached.
        
        # Let's generate start points.
        # Use a quasi-random sequence (Halton/Sobol) for good spread.
        # Simple Halton sequence for dim=5.
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        start_points = []
        for i in range(1, num_starts + 1):
            pt = np.array([halton(i, base) for base in [2, 3, 5, 7, 11]])
            # Map [0,1] to [-5,5]
            pt = pt * 10.0 - 5.0
            start_points.append(pt)
            
        # Shuffle starts? No, keep deterministic order for consistency, 
        # but maybe randomize which start gets which budget slice?
        # Let's just iterate.
        
        # We will run CMA-ES for each start.
        # CMA-ES parameters:
        # popsize: usually 4 + 3*ln(N) -> 4 + 3*ln(5) ~ 9. Let's use 10.
        # sigma: initial step size. 1.0 is reasonable for [-5,5].
        
        popsize = 10
        sigma = 1.0
        
        # We need to track evals carefully.
        # scipy.optimize.cma uses a callback or we can wrap func.
        
        def counting_func(x):
            nonlocal eval_count
            if eval_count >= budget:
                return np.inf
            eval_count += 1
            # Clip to bounds
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)
        
        # We need to handle the fact that CMA-ES might exceed budget.
        # We'll use maxiter to limit it, but also check eval_count inside.
        
        for i in range(num_starts):
            if eval_count >= budget:
                break
            if time.time() > deadline:
                break
                
            x0 = start_points[i]
            
            # Determine max iterations for this start
            # Remaining budget: budget - eval_count
            # CMA-ES uses roughly popsize evals per iteration.
            # So max_iter = (budget - eval_count) // popsize
            remaining = budget - eval_count
            max_iter = max(1, remaining // popsize)
            
            # To avoid one start eating all budget, we can cap max_iter per start
            # or just let it run until budget is gone.
            # If we let it run, early starts get more budget.
            # This is a "greedy" allocation.
            # Alternatively, fix budget per start: budget_per_start = budget // num_starts
            # Let's try fixed budget per start to ensure exploration.
            budget_per_start = budget // num_starts
            
            # If we are in the last start, give it all remaining budget?
            # No, strict allocation is safer for multi-start.
            # But if a start converges quickly, we waste budget.
            # CMA-ES stops if tol is reached.
            
            try:
                # Use scipy.optimize.cma
                # Note: scipy cma does not respect bounds directly in the same way, 
                # it uses a transformation or just lets it go out and we clip in counting_func.
                # Clipping in counting_func is fine.
                
                res = scipy.optimize.cma(
                    counting_func, 
                    x0, 
                    sigma0=sigma,
                    options={
                        'maxiter': max_iter,
                        'popsize': popsize,
                        'tol': 1e-8,
                        'verbose': 0
                    }
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception as e:
                # If CMA-ES fails, try Nelder-Mead from this start
                try:
                    res_nm = scipy.optimize.minimize(
                        counting_func, 
                        x0, 
                        method='Nelder-Mead',
                        options={'maxiter': max_iter * 2, 'xatol': 1e-8, 'fatol': 1e-8}
                    )
                    if res_nm.fun < best_f:
                        best_f = res_nm.fun
                        best_x = res_nm.x.copy()
                except Exception:
                    pass

        # If we still have budget left (e.g., CMA-ES converged early), 
        # use remaining for a local polish from the best point.
        if eval_count < budget and time.time() < deadline:
            remaining = budget - eval_count
            if remaining > 10:
                try:
                    res_polish = scipy.optimize.minimize(
                        counting_func, 
                        best_x, 
                        method='Nelder-Mead',
                        options={'maxiter': remaining // 2, 'xatol': 1e-10, 'fatol': 1e-10}
                    )
                    if res_polish.fun < best_f:
                        best_f = res_polish.fun
                        best_x = res_polish.x.copy()
                except Exception:
                    pass

        # Ensure we return something valid
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            eval_count += 1

        return best_x, best_f
