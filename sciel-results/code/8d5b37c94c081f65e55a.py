import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        lower, upper = -5.0, 5.0

        # Split budget: 60% for Global Search (ES), 40% for Local Refinement (NM)
        # Weierstrass f16 is multimodal. Global search finds basin, NM refines.
        es_budget = int(budget * 0.6)
        nm_budget = budget - es_budget

        # Ensure minimums
        if es_budget < 20:
            es_budget = max(1, budget // 2)
            nm_budget = budget - es_budget
        if nm_budget < 10 and budget > 20:
            nm_budget = 10
            es_budget = budget - nm_budget

        rng = np.random.default_rng(42)

        best_x_global = np.zeros(dim)
        best_f_global = np.inf

        # --- PHASE 1: (1+1)-ES Global Search ---
        # Start from a random point in the center
        x = rng.uniform(-4.0, 4.0, dim)
        fx = func(x)
        calls_used = 1
        
        if fx < best_f_global:
            best_f_global = fx
            best_x_global = x.copy()

        sigma = 2.0
        sigma_min = 0.01
        sigma_max = 5.0

        # 1/5 success rule parameters
        success_count = 0
        total_attempts = 0
        target_success_rate = 0.2

        for i in range(es_budget - 1):
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break

            # Generate offspring
            z = rng.standard_normal(dim)
            y = x + sigma * z
            y = np.clip(y, lower, upper)

            fy = func(y)
            calls_used += 1

            if fy < fx:
                x, fx = y, fy
                success_count += 1
                if fx < best_f_global:
                    best_f_global = fx
                    best_x_global = x.copy()
            else:
                # Keep current x, but update best if y is better than global best? 
                # No, (1+1) only moves if better. But y might be better than global best 
                # but worse than current x? Unlikely if x is local best, but possible 
                # if we had multiple restarts. Here single stream.
                if fy < best_f_global:
                    best_f_global = fy
                    best_x_global = y.copy()

            total_attempts += 1

            # Adapt sigma based on 1/5 rule
            if total_attempts >= 10:
                success_rate = success_count / total_attempts
                if success_rate > target_success_rate:
                    sigma *= 1.2
                elif success_rate < target_success_rate:
                    sigma *= 0.8
                
                # Reset counters
                success_count = 0
                total_attempts = 0

            sigma = np.clip(sigma, sigma_min, sigma_max)

        # --- PHASE 2: Nelder-Mead Local Refinement ---
        # Use the best point found in Phase 1 as the starting point for NM
        x_start = best_x_global.copy()
        
        # Create initial simplex for NM
        # Small initial step to ensure we are in the local basin
        simplex_step = 0.1 
        simplex = [x_start.copy()]
        for i in range(dim):
            p = x_start.copy()
            p[i] += simplex_step
            p = np.clip(p, lower, upper)
            simplex.append(p)
            
        # Wrapper for NM to count calls and enforce deadline
        nm_calls = 0
        nm_best_x = x_start.copy()
        nm_best_f = best_f_global
        
        def wrapped_func(x):
            nonlocal nm_calls, nm_best_x, nm_best_f, calls_used
            
            # Check budget and time
            if calls_used >= budget or time.time() > deadline:
                return 1e6
                
            x = np.clip(x, lower, upper)
            
            try:
                f_val = func(x)
            except Exception:
                f_val = 1e6
                
            calls_used += 1
            nm_calls += 1
            
            if np.isnan(f_val) or np.isinf(f_val):
                return 1e6
                
            # Update local best for NM
            if f_val < nm_best_f:
                nm_best_f = f_val
                nm_best_x = x.copy()
            
            return f_val

        def callback(xk):
            # Stop if we hit budget or time
            if calls_used >= budget or time.time() > deadline:
                return True
            return False

        try:
            # Run Nelder-Mead
            result = scipy.optimize.minimize(
                wrapped_func, 
                x_start, 
                method='Nelder-Mead',
                options={
                    'maxiter': nm_budget * 2, # Allow enough iterations
                    'xatol': 1e-8, 
                    'fatol': 1e-8, 
                    'adaptive': True,
                    'initial_simplex': simplex
                },
                callback=callback
            )
        except Exception:
            pass

        # Final result is the best found in either phase
        # nm_best_x/f is guaranteed to be <= best_x/f from ES
        # because NM started at best_x and only moves if it improves
        
        if nm_best_x is not None:
            return nm_best_x, nm_best_f
        else:
            return best_x_global, best_f_global
