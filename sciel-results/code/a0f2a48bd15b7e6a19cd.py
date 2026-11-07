"""Optimized BBOB f22 solver using Sobol global search and 1+1 CMA-ES refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize


class Optimizer:
    """Sobol-based global search with 1+1 CMA-ES local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget

        best_x = np.zeros(n)
        best_f = float('inf')
        calls = 0

        def try_eval(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            try:
                f = float(func(x))
            except Exception:
                f = float('inf')
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def powell_refine(x0, f0):
            """
            Refine x0 using Powell's method.
            """
            nonlocal calls, best_x, best_f, deadline
            
            if calls >= budget - 10 or time.time() > deadline - 10:
                return x0, f0

            local_best_x = x0.copy()
            local_best_f = f0
            
            def wrapped_func(x):
                nonlocal local_best_x, local_best_f, calls
                if calls >= budget or time.time() > deadline:
                    raise TimeoutError("Budget or deadline exceeded")
                
                x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
                try:
                    f = float(func(x))
                except Exception:
                    f = float('inf')
                calls += 1
                if f < local_best_f:
                    local_best_f = f
                    local_best_x = x.copy()
                return f

            try:
                # Powell's method is effective for low-dimensional problems
                res = minimize(wrapped_func, x0, method='Powell', options={'maxiter': 100, 'xtol': 1e-8, 'ftol': 1e-8})
                # Update global best if local best is better
                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = local_best_x.copy()
            except TimeoutError:
                pass
            except Exception:
                pass
            
            return local_best_x, local_best_f

        # Phase 1: Sobol quasi-random global sampling
        # Increased to 200 points for better coverage
        n_global = 200
        sobol_engine = qmc.Sobol(d=n, scramble=False)
        points = sobol_engine.random(n_global)
        points = points * 10.0 - 5.0
        
        global_candidates = []
        
        for i in range(n_global):
            if calls >= budget - 50 or time.time() > deadline - 30:
                break
            x = points[i]
            f = try_eval(x)
            if f is not None:
                global_candidates.append((f, x))

        # Sort global candidates by function value
        global_candidates.sort(key=lambda t: t[0])

        # Phase 2: Local refinement with Powell's method from best points
        # We try to refine the top few distinct candidates
        
        if not global_candidates:
             return best_x, best_f

        tried_x = []
        
        # Try refining the top 5 candidates if they are distinct
        for f_c, x_c in global_candidates[:5]:
            # Check distance to already tried points to ensure diversity
            is_far = True
            for tx in tried_x:
                if np.linalg.norm(x_c - tx) < 0.5:
                    is_far = False
                    break
            
            if not is_far:
                continue
                
            tried_x.append(x_c)
            
            # If we have enough budget left, run Powell
            if calls < budget - 20 and time.time() < deadline - 10:
                x_ref, f_ref = powell_refine(x_c, f_c)
                # powell_refine updates best_x/best_f if it finds something better

            # Stop if we've spent too much budget
            if calls > budget * 0.8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
