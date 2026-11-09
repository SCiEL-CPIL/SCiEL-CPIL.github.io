import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        dim = self.dim
        budget = self.budget

        # --- Phase 0: Initialize ---
        # Weierstrass f16 has a known structure.
        # BBOB f16: Weierstrass.
        # Periodicity and oscillations.
        # Let's do a smart initial scan + (1+1)-ES hybrid.
        
        # 1. Simple Grid/Random Scan to find promising regions
        # 2. (1+1)-ES with CMA-ES-like adaptation or simple 1/5 rule
        # 3. Local Nelder-Mead refinement
        
        # Strategy:
        # 1. Initial random sampling (10%) to get a good start.
        # 2. (1+1)-ES with 1/5 success rule (70%) to navigate the landscape.
        # 3. Nelder-Mead (20%) for final polishing.
        
        # Since dim=5, budget=500, we have plenty of room.
        # f16 Weierstrass is oscillatory. A global optimizer like CMA-ES or (1+1)-ES with large steps might struggle if steps are too small, but if too large, it's random walk.
        # (1+1)-ES with 1/5 success rule is robust.
        
        # Let's try a multi-start strategy if budget allows? No, budget is tight (500).
        # But 500 is enough for (1+1)-ES in 5D to find a good basin.
        
        # However, Weierstrass is multimodal. (1+1)-ES can get stuck in local minima.
        # Let's use a slightly modified (1+1)-ES that occasionally jumps (restarts) if stagnation is detected, or just rely on the step size adaptation to escape if it gets stuck?
        # Actually, (1+1)-ES with 1/5 rule increases step size on success, decreases on failure. If stuck, step size decreases to noise level. It won't jump out.
        # So, a "jumping" mechanism or multiple restarts is better for multimodal functions.
        
        # Given 500 budget:
        # Restart 1: 100 calls
        # Restart 2: 100 calls
        # Restart 3: 100 calls
        # Restart 4: 100 calls
        # Final Refinement: 100 calls on best so far?
        
        # Or:
        # Phase 1: Global Search (1+1)-ES with adaptive step size and occasional large jumps.
        # Phase 2: Local Refinement.
        
        # Let's try a "Multi-Start (1+1)-ES" approach.
        # Split budget into chunks. For each chunk, run (1+1)-ES starting from a random point or best-so-far perturbed.
        # Keep the best point found across all runs.
        # Final run: Nelder-Mead on the best point.
        
        # Better:
        # 1. Use 80% of budget for a robust global search.
        #    - Use (1+1)-ES with 1/5 success rule.
        #    - Add a "restart" mechanism: if no improvement for X iterations, jump to a new random point (or perturb best).
        # 2. Use 20% of budget for Nelder-Mead refinement on the best point.
        
        # Let's implement (1+1)-ES with restarts.
        
        calls_used = 0
        best_x = None
        best_f = float('inf')
        
        # Phase 1: Global Search
        # Budget for global search: 400 calls
        global_budget = 400
        # Phase 2: Local Refinement
        local_budget = budget - global_budget
        
        # Initial point: Center is often a good start for BBOB, but let's do a quick random scan to be safe.
        # Actually, random start is fine.
        
        x = np.random.uniform(-5, 5, dim)
        fx = func(x)
        calls_used += 1
        best_x = x.copy()
        best_f = fx
        
        sigma = 1.0
        no_improve_count = 0
        threshold = 10  # Restart after 10 non-improving steps? Maybe too aggressive.
        
        # 1/5 success rule parameters
        c_sigma = 1.0
        d_sigma = 1.0
        # Success rate target: 1/5
        # If success rate > 0.2, increase sigma.
        # If success rate < 0.2, decrease sigma.
        # We track a window of last 10 steps.
        successes = []
        
        for i in range(global_budget - 1):
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break
                
            # Check for stagnation
            # If no improvement for a long time, jump.
            if no_improve_count > 20:
                # Jump to a new random point
                x = np.random.uniform(-5, 5, dim)
                fx = func(x)
                calls_used += 1
                no_improve_count = 0
                successes = []
                sigma = 1.0
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                continue
            
            # Generate offspring
            z = np.random.standard_normal(dim)
            y = x + sigma * z
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls_used += 1
            
            # Update success history
            success = fy < fx
            successes.append(1 if success else 0)
            if len(successes) > 10:
                successes.pop(0)
            
            if success:
                no_improve_count = 0
                x = y
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                no_improve_count += 1
            
            # Adapt sigma using 1/5 success rule
            if len(successes) >= 5:
                success_rate = sum(successes) / len(successes)
                if success_rate > 0.2:
                    sigma *= 1.1
                else:
                    sigma /= 1.1
                
                # Bounds for sigma
                sigma = np.clip(sigma, 0.01, 10.0)
        
        # Phase 2: Local Refinement with Nelder-Mead
        # Use the best point found so far
        x0 = best_x.copy()
        
        # Wrapper to count calls
        nm_calls = 0
        nm_best_x = x0.copy()
        nm_best_f = best_f
        
        def wrapped_func(x):
            nonlocal nm_calls, nm_best_x, nm_best_f, calls_used
            if calls_used >= budget or time.time() > deadline:
                return 1e6
            x = np.clip(x, -5, 5)
            try:
                f_val = func(x)
            except:
                f_val = 1e6
            calls_used += 1
            nm_calls += 1
            if f_val < nm_best_f:
                nm_best_f = f_val
                nm_best_x = x.copy()
            return f_val
        
        def callback(xk):
            if calls_used >= budget or time.time() > deadline:
                return True
            return False
        
        try:
            # Nelder-Mead
            # Initial simplex: perturb x0 slightly
            simplex = [x0.copy()]
            for i in range(dim):
                p = x0.copy()
                p[i] += 0.1
                simplex.append(p)
                
            minimize(
                wrapped_func,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': local_budget * 2,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True,
                    'initial_simplex': simplex
                },
                callback=callback
            )
        except:
            pass
        
        # Return the best found in either phase
        if nm_best_f < best_f:
            return nm_best_x, nm_best_f
        else:
            return best_x, best_f
