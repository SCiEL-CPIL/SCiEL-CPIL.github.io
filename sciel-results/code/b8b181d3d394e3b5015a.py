import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = time.time()
        # Safety margin: 330s limit, use 310s to be safe
        self.deadline = self.start_time + 310

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        # Strategy: Multi-Start Nelder-Mead
        # Griewank-Rosenbrock has many local minima.
        # We use a fraction of the budget for a global "scatter" phase to find promising basins,
        # then use Nelder-Mead to converge.
        
        # We will define a helper for Nelder-Mead that respects the budget and time limits.
        def nelder_mead(x0, f0, max_iter, func, calls_ref, time_limit):
            """
            Custom Nelder-Mead implementation.
            x0: initial point
            f0: function value at x0
            max_iter: max iterations
            func: objective function
            calls_ref: list containing [current_calls] to track global budget
            time_limit: deadline timestamp
            """
            dim = len(x0)
            # Initialize simplex
            # Start with x0 and small perturbations
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            
            # Initial step size
            # For [-5, 5], a step of 0.1 or 0.5 is reasonable for NM.
            # Let's use 0.1 for fine-tuning after a global search, or 0.5 if starting fresh.
            # Since we are doing multi-start, we want to explore locally.
            step = 0.5
            
            for i in range(1, dim + 1):
                simplex[i] = x0.copy()
                simplex[i][i-1] += step
                
            # Evaluate initial simplex
            fvals = np.zeros(dim + 1)
            fvals[0] = f0
            
            for i in range(1, dim + 1):
                # Check budget
                if calls_ref[0] >= budget:
                    return x0, f0
                if time.time() > time_limit:
                    return x0, f0
                
                fvals[i] = func(simplex[i])
                calls_ref[0] += 1
                
                # Update global best
                if fvals[i] < best_f:
                    nonlocal best_f, best_x
                    best_f = fvals[i]
                    best_x = simplex[i].copy()
            
            # Nelder-Mead Iterations
            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction
            sigma = 0.5   # Shrinkage
            
            for _ in range(max_iter):
                if calls_ref[0] >= budget:
                    break
                if time.time() > time_limit:
                    break
                
                # Sort simplex by function value
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]
                
                x_best = simplex[0]
                x_worst = simplex[-1]
                x_second_worst = simplex[-2]
                
                # Check convergence
                # If the spread of function values is very small, stop
                if np.max(fvals) - np.min(fvals) < 1e-8:
                    break
                
                # Centroid of all but the worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # 1. Reflection
                x_r = centroid + alpha * (centroid - x_worst)
                # Clip to bounds
                x_r = np.clip(x_r, -5, 5)
                
                if calls_ref[0] >= budget: break
                if time.time() > time_limit: break
                
                f_r = func(x_r)
                calls_ref[0] += 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()
                    
                # 2. Expansion
                if f_r < fvals[0]:
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    
                    if calls_ref[0] >= budget: break
                    if time.time() > time_limit: break
                    
                    f_e = func(x_e)
                    calls_ref[0] += 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                        
                # 3. Contraction
                elif f_r < fvals[-2]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    
                    if calls_ref[0] >= budget: break
                    if time.time() > time_limit: break
                    
                    f_c = func(x_c)
                    calls_ref[0] += 1
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()
                    
                    if f_c < f_r:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # 4. Shrinkage
                        for i in range(1, dim + 1):
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            
                            if calls_ref[0] >= budget: break
                            if time.time() > time_limit: break
                            
                            fvals[i] = func(simplex[i])
                            calls_ref[0] += 1
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()
                            if calls_ref[0] >= budget: break
                            if time.time() > time_limit: break
                            if calls_ref[0] >= budget: break
                            if time.time() > time_limit: break
                        if calls_ref[0] >= budget: break
                        if time.time() > time_limit: break
                        simplex[0] = x_best
                        fvals[0] = fvals[0] # best value doesn't change in shrinkage relative to best
                        # Actually, x_best is simplex[0] before shrinkage.
                        # After shrinkage, simplex[0] is still x_best.
                        # We re-evaluated others.
                        # We need to re-sort or just continue.
                        
                else:
                    # 4. Shrinkage (same as above)
                    for i in range(1, dim + 1):
                        simplex[i] = x_best + sigma * (simplex[i] - x_best)
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        
                        if calls_ref[0] >= budget: break
                        if time.time() > time_limit: break
                        
                        fvals[i] = func(simplex[i])
                        calls_ref[0] += 1
                        if fvals[i] < best_f:
                            best_f = fvals[i]
                            best_x = simplex[i].copy()
                        if calls_ref[0] >= budget: break
                        if time.time() > time_limit: break
                    if calls_ref[0] >= budget: break
                    if time.time() > time_limit: break
                    
            return x_best, fvals[0]

        # Main Loop: Multi-Start
        # We divide the budget.
        # Let's try 10 restarts.
        # Each restart gets budget/10 calls.
        # 500 / 10 = 50 calls per restart.
        # NM needs at least dim+1 = 6 calls to start.
        # 50 calls is enough for a few NM steps.
        
        num_restarts = 10
        budget_per_restart = budget // num_restarts
        
        calls_ref = [0]
        
        for r in range(num_restarts):
            if calls_ref[0] >= budget:
                break
            if time.time() > self.deadline:
                break
                
            # Generate initial point
            # Use a stratified or random initialization.
            # Random is fine.
            x0 = rng.uniform(-5, 5, dim)
            
            if calls_ref[0] >= budget:
                break
            if time.time() > self.deadline:
                break
            
            f0 = func(x0)
            calls_ref[0] += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            
            # Run Nelder-Mead with remaining budget for this restart
            # We pass the global calls_ref so it stops when global budget is hit.
            # max_iter is just a safety cap on the number of NM iterations.
            # Each iteration uses 1 to dim+1 calls.
            # Let's allow up to budget_per_restart iterations, but it will stop earlier due to calls.
            max_iter = budget_per_restart + dim # Safety
            
            # Note: nelder_mead uses nonlocal best_f, best_x.
            # This is valid inside __call__.
            nelder_mead(x0, f0, max_iter, func, calls_ref, self.deadline)
            
        return best_x, best_f
