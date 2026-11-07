import numpy as np
import scipy.optimize as opt
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s limit, stop at 290s to allow for cleanup/return
        deadline = start_time + 290.0
        
        rng = np.random.default_rng(123)
        
        best_x = None
        best_f = np.inf
        used_calls = 0
        max_calls = self.budget
        
        def safe_func(x):
            nonlocal used_calls
            if used_calls >= max_calls:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("Time limit exceeded")
            used_calls += 1
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)
        
        # Phase 1: Global Exploration with Sobol Sequence
        # Use a moderate number of initial points to balance exploration and exploitation
        n_initial = 30
        n_initial = min(n_initial, max_calls // 3)
        
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        sobol_points = sampler.random(n_initial)
        initial_points = sobol_points * 10 - 5
        
        initial_values = []
        
        for p in initial_points:
            if used_calls >= max_calls or time.time() > deadline:
                break
            try:
                fp = safe_func(p)
                initial_values.append((fp, p))
                if fp < best_f:
                    best_f = fp
                    best_x = p.copy()
            except RuntimeError:
                break
        
        if best_x is None:
            return None, np.inf

        # Phase 2: Multi-Start Local Optimization with Nelder-Mead
        initial_values.sort(key=lambda x: x[0])
        
        remaining_budget = max_calls - used_calls
        
        if remaining_budget > 10:
            # Determine number of restarts
            # Each NM run needs at least dim+1 calls to form simplex, plus iterations
            # Let's aim for at most 8 restarts, or as many as budget allows
            max_restarts = 8
            budget_per_restart = remaining_budget // max_restarts
            
            # If budget per restart is too small, reduce restarts
            if budget_per_restart < 25:
                max_restarts = max(1, remaining_budget // 25)
                budget_per_restart = remaining_budget // max_restarts
            
            # Select top candidates from initial_values
            # If we have fewer initial points than max_restarts, use all
            candidates = initial_values[:max_restarts]
            
            # If we have more restarts allowed than candidates, add some random points
            # to ensure diversity if initial exploration was poor
            while len(candidates) < max_restarts and used_calls + 1 < max_calls:
                x_rand = rng.uniform(-5, 5, self.dim)
                try:
                    fp_rand = safe_func(x_rand)
                    candidates.append((fp_rand, x_rand))
                    if fp_rand < best_f:
                        best_f = fp_rand
                        best_x = x_rand.copy()
                except RuntimeError:
                    break
            
            # Sort candidates again by function value
            candidates.sort(key=lambda x: x[0])
            
            for fp, x_start in candidates:
                if used_calls >= max_calls or time.time() > deadline:
                    break
                
                local_budget = max_calls - used_calls
                if local_budget < 15:
                    break
                
                try:
                    # Use Nelder-Mead with adaptive parameters
                    result = opt.minimize(
                        safe_func, 
                        x_start, 
                        method='Nelder-Mead',
                        options={
                            'maxiter': local_budget, 
                            'maxfev': local_budget,
                            'xatol': 1e-10,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                    
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                        
                except (RuntimeError, Exception):
                    # If budget or time exceeded, stop further restarts
                    if time.time() > deadline or used_calls >= max_calls:
                        break
                    # If it was just a convergence failure, continue to next restart
                    continue
        
        # Phase 3: Final Polish (if budget remains)
        # If we still have significant budget left, do a final NM run from the global best
        if used_calls < max_calls and time.time() < deadline:
            final_budget = max_calls - used_calls
            if final_budget > 15:
                try:
                    result = opt.minimize(
                        safe_func, 
                        best_x, 
                        method='Nelder-Mead',
                        options={
                            'maxiter': final_budget,
                            'maxfev': final_budget,
                            'xatol': 1e-12,
                            'fatol': 1e-12,
                            'adaptive': True
                        }
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except (RuntimeError, Exception):
                    pass

        return best_x, best_f
