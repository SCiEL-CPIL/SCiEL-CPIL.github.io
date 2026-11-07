import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        d = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        used_calls = 0

        def check_time():
            return time.time() > deadline

        def try_point(x):
            nonlocal best_x, best_f, used_calls
            if used_calls >= budget or check_time():
                return
            try:
                fx = func(x)
            except Exception:
                return
            used_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 1: Global exploration using Sobol sequence
        # We use a fixed number of Sobol points to ensure good coverage.
        # 5 dimensions, 2^7 = 128 points is a good start for 500 budget.
        # Let's use 128 points for exploration.
        num_sobol = 128
        
        # Generate Sobol points in [0, 1]^d
        sampler = qmc.Sobol(d=d, scramble=True)
        # We need to generate 2^k points. 128 is 2^7.
        # scipy's Sobol requires n to be a power of 2 for standard usage, 
        # or we can just generate a larger power and take the first N.
        # Let's generate 128 points directly.
        sobol_points = sampler.random(n=num_sobol)
        
        # Map from [0, 1] to [-5, 5]
        # x = -5 + 10 * u
        sobol_points_mapped = -5.0 + 10.0 * sobol_points
        
        for x in sobol_points_mapped:
            if check_time() or used_calls >= budget:
                break
            try_point(x)
            
        # If we didn't use all Sobol points, we could continue, but let's switch to local.
        # If best_x is None, something went wrong, but try_point should have set it.
        if best_x is None:
            # Fallback: try origin
            try_point(np.zeros(d))
            if best_x is None:
                return None, np.inf

        # Phase 2: Local refinement using Nelder-Mead
        remaining_budget = budget - used_calls
        if remaining_budget > 10 and not check_time():
            x0 = best_x.copy()
            
            # Wrap func to count calls and enforce limits
            calls_made = [0]
            max_local_calls = remaining_budget
            
            def wrapped_func(x):
                calls_made[0] += 1
                if calls_made[0] > max_local_calls or check_time():
                    return 1e10  # Large penalty to stop optimization
                try:
                    return func(x)
                except Exception:
                    return 1e10

            # Nelder-Mead options
            # xatol and fatol control convergence.
            # maxiter should be bounded by remaining calls.
            # Each NM iteration uses at most d+1 function evaluations? 
            # Actually, it's more complex, but maxiter limits the number of iterations.
            # A safe maxiter is remaining_budget // (d+1) + 10
            max_iter = max(10, remaining_budget // (d + 1) + 10)
            
            try:
                res = minimize(wrapped_func, x0, method='Nelder-Mead',
                               options={'xatol': 1e-8, 'fatol': 1e-10,
                                        'maxiter': max_iter,
                                        'adaptive': True,
                                        'initial_simplex': None}) # Let scipy choose, or we can specify
                # Update best if NM found something better
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Final fine-tuning if budget remains
        # Sometimes NM stops a bit early or gets stuck. A few random small steps might help.
        if best_x is not None and used_calls < budget and not check_time():
            rng = np.random.default_rng(42)
            x = best_x.copy()
            fx = best_f
            remaining = budget - used_calls
            
            # Try decreasing scales
            for scale in [0.1, 0.01, 0.001]:
                if used_calls >= budget or check_time():
                    break
                # Use a fraction of remaining budget
                num_steps = min(20, remaining)
                for _ in range(num_steps):
                    if used_calls >= budget or check_time():
                        break
                    z = rng.standard_normal(d)
                    y = np.clip(x + scale * z, -5, 5)
                    try_point(y)
                    if best_x is not None and best_f < fx:
                        x = best_x.copy()
                        fx = best_f

        return best_x, best_f
