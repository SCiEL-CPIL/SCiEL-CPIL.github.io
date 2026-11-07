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
        deadline = start_time + 325  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls = 0

        def wrapper(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return 1e9
            if calls >= budget:
                return 1e9
            calls += 1
            x = np.clip(x, -5, 5)
            try:
                f = float(func(x))
            except:
                f = 1e9

            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Initial Space-Filling Sampling
        # Use Sobol sequence for deterministic, well-distributed initial points
        # 50 points is a good trade-off: enough to find a basin, not too many to waste budget
        n_init = min(50, budget)
        
        try:
            sampler = qmc.Sobol(d=dim, seed=42)
            points = sampler.random(n_init)
            # Map from [0, 1]^d to [-5, 5]^d
            points = points * 10.0 - 5.0
        except Exception:
            # Fallback to random if Sobol fails
            points = np.random.uniform(-5, 5, (n_init, dim))

        # Evaluate initial points
        initial_results = []
        for pt in points:
            if calls >= budget or time.time() > deadline:
                break
            f_val = wrapper(pt)
            initial_results.append((f_val, pt))
            
        if not initial_results:
            return best_x, best_f

        # Sort initial results to find the best candidates
        initial_results.sort(key=lambda x: x[0])
        
        # Phase 2: Multi-start Nelder-Mead Refinement
        # Select top M candidates for local refinement
        # M=5 is a good heuristic for dim=5, budget=500
        # We allocate the remaining budget to these M starts
        remaining_budget = budget - calls
        if remaining_budget <= 0 or best_x is None:
            return best_x, best_f

        # Choose number of restarts based on remaining budget
        # If we have very little budget, just do 1 or 2.
        # If we have lots, we can do more, but NM is fast so 5 is usually enough.
        n_restarts = min(5, len(initial_results))
        
        # Distribute remaining budget among restarts
        # Reserve a tiny bit for safety? No, NM stops when maxiter is hit.
        # We need to estimate maxiter for NM.
        # NM in 5D typically uses a simplex of size 6.
        # Each iteration involves ~6 function evaluations (worst case).
        # So maxiter ~ budget / 6.
        
        budget_per_restart = remaining_budget // n_restarts
        
        # If budget_per_restart is too small, reduce restarts
        if budget_per_restart < 10:
            n_restarts = max(1, remaining_budget // 10)
            budget_per_restart = remaining_budget // n_restarts

        # Perform local optimization from the best starting points
        for i in range(n_restarts):
            if time.time() > deadline or calls >= budget:
                break
            
            start_pt = initial_results[i][1]
            
            # Calculate maxiter for Nelder-Mead
            # Each iteration of NM in N dimensions can take up to N+1 evaluations
            # To be safe, we assume ~6 evaluations per iteration for dim=5
            maxiter = max(1, int(budget_per_restart / 6))
            
            try:
                res = minimize(
                    wrapper, 
                    start_pt, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': maxiter,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        # Phase 3: Final Polish
        # If there is still budget left and we have a good best_x, do a final tight polish
        remaining_budget = budget - calls
        if remaining_budget > 5 and best_x is not None:
            final_maxiter = max(1, remaining_budget // 6)
            try:
                res_final = minimize(
                    wrapper, 
                    best_x, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': final_maxiter,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        return best_x, best_f
