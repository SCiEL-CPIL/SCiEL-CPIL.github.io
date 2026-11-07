import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        max_calls = self.budget
        calls_made = 0
        dim = self.dim

        # Wrapper for counting and clipping
        def eval_func(x):
            nonlocal calls_made
            if calls_made >= max_calls:
                return np.inf
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            calls_made += 1
            return val

        best_x = np.zeros(dim)
        best_f = np.inf

        # Structural Change: Multi-Start L-BFGS-B
        # Rosenbrock is smooth and gradient-based methods are effective once in the valley.
        # We use diverse starts to ensure we find the valley.
        
        # Generate diverse starting points
        # 1. Origin
        # 2. Corners of the box (scaled)
        # 3. Random points
        
        rng = np.random.default_rng(42)
        
        # Pre-defined diverse starts
        starts = []
        starts.append(np.zeros(dim))
        
        # Add some structured points
        # The minimizer is in [-4, 4]^5. 
        # Rosenbrock global min is at (1,1,...,1). 
        # Hidden shift/rotation means we don't know where (1,1) maps to.
        # So we need broad coverage.
        
        # Let's use a grid-like or random spread.
        # 5D space is small. 500 calls is plenty for 10-20 L-BFGS runs.
        
        n_starts = 15
        
        # Add random points
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4.5, 4.5, dim))
            
        # Sort starts by distance to origin? No, random is fine.
        # Maybe prioritize points closer to origin if we assume shift is small? 
        # The contract says shift is hidden. 
        # Let's just use the random ones plus origin.
        
        # We will run L-BFGS-B from each start.
        # We need to track the global best across all runs.
        
        # Allocate budget per start roughly
        # L-BFGS-B in 5D is fast. 20-50 evals is typical for convergence.
        # 500 / 15 ~ 33 evals per start.
        
        for start in starts:
            if time.time() > deadline:
                break
            if calls_made >= max_calls:
                break
                
            # Local best for this start
            local_best_x = start.copy()
            local_best_f = eval_func(local_best_x)
            
            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()
                
            # Run L-BFGS-B
            # We pass the current best as the starting point for the optimizer?
            # No, L-BFGS-B starts from `start`.
            
            # We need a wrapper that updates global best?
            # L-BFGS-B calls the function multiple times.
            # We can just use the standard wrapper.
            
            # To avoid wasting budget if L-BFGS-B fails to improve,
            # we can check if the first step is promising? 
            # Or just let it run.
            
            try:
                # maxiter is not strictly an eval count, but usually 1-2 evals per iter.
                # We want to limit total evals.
                # SciPy's L-BFGS-B doesn't have a strict 'max_evals' option in older versions,
                # but 'maxiter' is a good proxy.
                # In 5D, 50 iterations is usually enough.
                # 50 iters * 1-2 evals = 50-100 evals.
                # This might exceed our per-start budget if we have 15 starts.
                # Let's reduce maxiter to 20-30.
                
                res = minimize(eval_func, start, method='L-BFGS-B', 
                               bounds=[(-5, 5)] * dim,
                               options={'maxiter': 20, 'ftol': 1e-8, 'gtol': 1e-5})
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass

        # Polish with Nelder-Mead if time/budget remains
        if calls_made < max_calls and time.time() < deadline:
            nm_calls = 0
            nm_limit = max_calls - calls_made
            
            def nm_func(z):
                nonlocal nm_calls
                if nm_calls >= nm_limit:
                    return np.inf
                nm_calls += 1
                return func(np.clip(z, -5, 5))
            
            try:
                max_iter = max(1, nm_limit // 6)
                res = minimize(nm_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
