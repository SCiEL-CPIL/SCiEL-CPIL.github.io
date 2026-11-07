import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0  # Safety margin against 330s limit
        
        def time_check():
            return (time.time() - start_time) > timeout

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)
        
        best_x = None
        best_f = np.inf
        eval_count = 0
        
        # Strategy: Multi-Start Nelder-Mead
        # 1. Use a portion of the budget to find good initial points via random sampling.
        # 2. Use the remaining budget to perform Nelder-Mead local optimization from the best points.
        
        # Phase 1: Global Exploration
        # Allocate ~20% of budget for random sampling to find diverse basins.
        # For budget=500, that's 100 points.
        n_global = max(10, budget // 5)
        
        # Ensure we don't exceed budget
        n_global = min(n_global, budget)
        
        # Sample points
        # We want points spread out. Uniform in [-5, 5] is good, but maybe bias slightly to center?
        # The minimizer is in [-4, 4]. Uniform [-5, 5] is safe.
        candidates = []
        
        for i in range(n_global):
            if eval_count >= budget or time_check():
                break
            x = rng.uniform(-5.0, 5.0, dim)
            f = func(x)
            eval_count += 1
            candidates.append((f, x))
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        if eval_count >= budget or time_check():
            return best_x, best_f
        
        # Sort candidates by fitness
        candidates.sort(key=lambda x: x[0])
        
        # Phase 2: Local Refinement with Nelder-Mead
        # Select top K candidates to refine.
        # How many? 
        # Each Nelder-Mead run in dim=5 takes roughly 2 * dim * iterations evaluations.
        # If we allow ~50-100 evals per NM run, we can do 3-5 runs.
        # Let's pick the top 3 unique-ish points (simple distance check or just top 3).
        
        max_nm_runs = 5
        nm_budget_per_run = (budget - eval_count) // max_nm_runs
        
        # If nm_budget_per_run is too small, reduce runs
        if nm_budget_per_run < 10:
             max_nm_runs = 1
             nm_budget_per_run = budget - eval_count
             
        # Select starting points
        # We just take the top min(max_nm_runs, len(candidates))
        start_points = [c[1] for c in candidates[:min(max_nm_runs, len(candidates))]]
        
        # Nelder-Mead Implementation
        def nelder_mead(func, x0, max_iter, time_check, budget_remaining, eval_count_ref):
            # x0: initial point (numpy array)
            # We need to manage eval_count and best_x/best_f externally or via closure.
            # Let's use a local counter and update the outer scope at the end? 
            # No, we need to track global best.
            # Let's pass a mutable object or just return the best found and let the caller update.
            # But we need to update global best_x/best_f as we go to be safe? 
            # Actually, NM finds a local min. We can just run it and update global best at the end of each run.
            # However, if NM fails or stagnates, we might waste budget.
            # Let's implement a lightweight NM that checks time/budget every step.
            
            dim = len(x0)
            # Initialize simplex
            # Standard NM: n+1 points
            # Initial step size: small, e.g., 0.1 or relative to domain
            step_size = 0.1
            simplex = [x0.copy() for _ in range(dim + 1)]
            # Perturb the simplex
            for i in range(dim):
                simplex[i+1][i] += step_size
            
            f_values = [np.inf] * (dim + 1)
            # Evaluate initial simplex
            for i in range(dim + 1):
                if eval_count_ref[0] >= budget_remaining or time_check():
                    return None, None, eval_count_ref[0]
                f_values[i] = func(simplex[i])
                eval_count_ref[0] += 1
                
            best_idx = 0
            best_f_local = f_values[0]
            best_x_local = simplex[0].copy()
            
            for it in range(max_iter):
                if time_check() or eval_count_ref[0] >= budget_remaining:
                    break
                    
                # Sort simplex by f_values
                sorted_indices = np.argsort(f_values)
                sorted_simplex = [simplex[i] for i in sorted_indices]
                sorted_f = [f_values[i] for i in sorted_indices]
                
                # Reassign
                for i in range(dim + 1):
                    simplex[i] = sorted_simplex[i]
                    f_values[i] = sorted_f[i]
                    
                # Check convergence
                # If the range of f values is small, stop
                if abs(f_values[-1] - f_values[0]) < 1e-8:
                    break
                    
                # Centroid of all but worst point
                centroid = np.mean(simplex[:dim], axis=0)
                
                # Reflection
                alpha = 1.0
                x_r = centroid + alpha * (centroid - simplex[-1])
                if eval_count_ref[0] >= budget_remaining or time_check():
                    break
                f_r = func(x_r)
                eval_count_ref[0] += 1
                
                if f_values[0] <= f_r < f_values[-2]:
                    # Replace worst with reflection
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                elif f_r < f_values[0]:
                    # Expansion
                    gamma = 2.0
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    if eval_count_ref[0] >= budget_remaining or time_check():
                        break
                    f_e = func(x_e)
                    eval_count_ref[0] += 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                else:
                    # Contraction
                    rho = 0.5
                    # Inside contraction if f_r >= f_values[-1], else outside
                    if f_r < f_values[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (x_r - centroid)
                    if eval_count_ref[0] >= budget_remaining or time_check():
                        break
                    f_c = func(x_c)
                    eval_count_ref[0] += 1
                    if f_c < min(f_r, f_values[-1]):
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            if eval_count_ref[0] >= budget_remaining or time_check():
                                break
                            f_values[i] = func(simplex[i])
                            eval_count_ref[0] += 1
                        if time_check() or eval_count_ref[0] >= budget_remaining:
                            break
                            
                # Update local best
                min_idx = np.argmin(f_values)
                if f_values[min_idx] < best_f_local:
                    best_f_local = f_values[min_idx]
                    best_x_local = simplex[min_idx].copy()
                    
            return best_x_local, best_f_local, eval_count_ref[0]

        # Run Nelder-Mead on selected starts
        # We need a mutable counter for eval_count to pass to NM
        # But NM is defined to take eval_count_ref as a list [count]
        
        for start_point in start_points:
            if eval_count >= budget or time_check():
                break
            
            # Determine max_iter for NM
            # Roughly, each iteration uses 1-3 evals.
            # Let's set max_iter such that 3 * max_iter <= nm_budget_per_run
            # But we also have time limits.
            # Let's just let the time_check and budget check inside NM handle it.
            # Set a high max_iter and rely on budget/time.
            max_iter = 1000 
            
            # Pass a reference to eval_count
            # Since we can't pass int by reference, we use a list
            eval_ref = [eval_count]
            
            res_x, res_f, new_count = nelder_mead(func, start_point, max_iter, time_check, budget, eval_ref)
            eval_count = new_count
            
            if res_f is not None and res_f < best_f:
                best_f = res_f
                best_x = res_x.copy()
                
        return best_x, best_f
