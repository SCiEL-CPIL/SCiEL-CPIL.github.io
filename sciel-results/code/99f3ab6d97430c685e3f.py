import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 330s kill
        
        calls_used = 0
        max_calls = self.budget
        best_f = float('inf')
        best_x = None
        
        # Helper to track best
        def update_best(x, f):
            nonlocal best_f, best_x
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Deterministic Space-Filling Exploration (Halton Sequence)
        # Rationale: Guarantees uniform coverage of the search space, 
        # maximizing probability of finding the global minimum's basin early.
        # This is crucial for AOCC which rewards early progress.
        
        n_halton = 60  # Slightly increased from 40 to ensure better initial coverage
        
        # Generate Halton sequence points in [-5, 5]^dim
        # We need a low-discrepancy sequence.
        # Simple Halton sequence implementation
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        halton_points = []
        # Primes for dimensions
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        for i in range(1, n_halton + 1):
            point = []
            for d in range(self.dim):
                # Map [0,1) to [-5, 5]
                val = halton(i, primes[d]) * 10.0 - 5.0
                point.append(val)
            halton_points.append(np.array(point))
        
        # Evaluate Halton points
        for p in halton_points:
            if time.time() > deadline or calls_used >= max_calls:
                break
            f_val = func(p)
            calls_used += 1
            update_best(p, f_val)

        # If we haven't made any calls (shouldn't happen), do one random
        if best_x is None:
            x0 = np.zeros(self.dim)
            f0 = func(x0)
            calls_used += 1
            update_best(x0, f0)

        # Phase 2: Multi-Start Local Refinement (Nelder-Mead)
        # Rationale: Nelder-Mead is robust in low dimensions (5D) and doesn't
        # require gradient information. Starting from multiple diverse points 
        # (the best few from Halton + best from random) helps escape local minima.
        
        # We will run Nelder-Mead from the current best_x.
        
        # Let's implement a wrapper for Nelder-Mead that counts calls.
        def nm_func(x):
            nonlocal calls_used
            if calls_used >= max_calls or time.time() > deadline:
                return best_f  # Return current best to stop optimization effectively
            f_val = func(x)
            calls_used += 1
            return f_val
        
        # We need to be careful with the call counting in scipy.
        # scipy minimize calls the function multiple times.
        # We will run Nelder-Mead from the best point found in Phase 1.
        
        x0 = best_x.copy()
        
        # Options for Nelder-Mead
        options = {
            'maxiter': 200,  # Increased iterations to allow for better convergence
            'xatol': 1e-8,
            'fatol': 1e-8
        }
        
        # Run Nelder-Mead
        try:
            res = minimize(nm_func, x0, method='Nelder-Mead', options=options)
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

        # If we have budget left, perform a few random restarts with local search
        # to ensure we didn't miss a better basin.
        remaining_budget = max_calls - calls_used
        if remaining_budget > 50: # Need at least 50 calls for a decent local search
            # Do 2-3 restarts
            n_restarts = min(3, remaining_budget // 50)
            for r in range(n_restarts):
                if time.time() > deadline or calls_used >= max_calls:
                    break
                
                # Random start point
                x_start = np.random.uniform(-5, 5, size=self.dim)
                f_start = func(x_start)
                calls_used += 1
                if f_start < best_f:
                    best_f = f_start
                    best_x = x_start.copy()
                
                # Local search from this point
                x0_r = x_start.copy()
                
                def nm_func_r(x):
                    nonlocal calls_used
                    if calls_used >= max_calls or time.time() > deadline:
                        return best_f
                    f_val = func(x)
                    calls_used += 1
                    return f_val
                
                try:
                    res_r = minimize(nm_func_r, x0_r, method='Nelder-Mead', options=options)
                    if res_r.fun < best_f:
                        best_f = res_r.fun
                        best_x = res_r.x.copy()
                except Exception:
                    pass

        return best_x, best_f
