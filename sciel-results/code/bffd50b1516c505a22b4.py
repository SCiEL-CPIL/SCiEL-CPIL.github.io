import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin
        dim = self.dim
        budget = self.budget
        
        call_count = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        candidates = [] # List of (x, f)

        def do_call(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
                
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            call_count += 1
            candidates.append((x_clipped, val))
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        # Phase 1: Global Sampling with Sobol
        # Use 50 points for initial exploration
        num_global = 50
        if num_global > budget:
            num_global = budget
            
        if num_global > 0:
            sampler = qmc.Sobol(d=dim, scramble=True)
            sobol_points = sampler.random(num_global) * 10 - 5
            for p in sobol_points:
                if call_count >= budget or time.time() > deadline:
                    break
                do_call(p)
        
        # If we ran out of budget in Phase 1, return best
        if call_count >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Multi-Start Nelder-Mead
        # Sort candidates by fitness
        candidates.sort(key=lambda x: x[1])
        
        # Select top 5 distinct candidates for local refinement
        # Ensure we have at least 5 candidates, if not, use all
        num_starts = min(5, len(candidates))
        
        # Allocate remaining budget to these starts
        remaining_budget = budget - call_count
        # Leave a small buffer for safety
        alloc_per_start = max(10, remaining_budget // num_starts)
        
        for i in range(num_starts):
            if call_count >= budget or time.time() > deadline:
                break
                
            x0 = candidates[i][0]
            
            # Create a wrapper for NM to track calls and enforce budget
            nm_calls = [0]
            max_nm_calls = alloc_per_start
            
            def nm_func(x, nm_calls=nm_calls, max_nm_calls=max_nm_calls):
                nonlocal call_count, best_f, best_x
                if nm_calls[0] >= max_nm_calls:
                    return best_f
                if call_count >= budget:
                    return best_f
                if time.time() > deadline:
                    return best_f
                    
                nm_calls[0] += 1
                return do_call(x)
            
            try:
                # Use Nelder-Mead with adaptive simplex
                minimize(nm_func, x0, method='Nelder-Mead',
                         options={'maxiter': max_nm_calls, 'xatol': 1e-8, 'fatol': 1e-8, 
                                  'adaptive': True, 'maxfev': max_nm_calls})
            except Exception:
                pass
        
        return best_x, best_f
