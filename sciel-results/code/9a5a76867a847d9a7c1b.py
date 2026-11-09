import numpy as np
import time
import itertools

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit
        
        n = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        def try_eval(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= budget:
                return
            if time.time() > deadline:
                return
            try:
                fx = func(x)
            except:
                return
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 1: Coarse Global Search with Lattice
        # Use a 3^5 = 243 point grid covering [-5, 5]^5
        # Points: -5, 0, 5
        grid_vals = np.array([-5.0, 0.0, 5.0])
        
        # Generate grid points
        for combo in itertools.product(grid_vals, repeat=n):
            x = np.array(combo)
            try_eval(x)
            if calls_used >= budget or time.time() > deadline:
                break
        
        if best_x is None:
            # Fallback if no evaluation succeeded (shouldn't happen)
            x0 = np.zeros(n)
            try_eval(x0)
            if best_x is None:
                best_x = x0
                best_f = 0.0
        
        # Phase 2: CMA-ES Refinement
        # Use a simple implementation of CMA-ES or a robust (1+1)-ES with restarts
        # Given the budget (500 total, ~250 remaining), we can afford a decent population.
        # Let's use a simple population-based ES with adaptive step size.
        
        remaining = budget - calls_used
        if remaining > 0:
            # Initialize population around best_x
            # Use a moderate initial sigma
            sigma = 1.0
            
            # Simple (1+1)-ES with 1/5 rule is robust but slow.
            # Let's try a small population CMA-ES-like approach or just a well-tuned (1+1)-ES.
            # For 5D, (1+1)-ES is often competitive if step size adaptation is good.
            
            x = best_x.copy()
            f = best_f
            rng = np.random.default_rng(42)
            
            # (1+1)-ES with 1/5 rule
            while calls_used < budget:
                if time.time() > deadline:
                    break
                
                # Generate mutation
                z = rng.standard_normal(n)
                x_new = x + sigma * z
                x_new = np.clip(x_new, -5, 5)
                
                f_new = func(x_new)
                calls_used += 1
                
                if f_new < f:
                    x = x_new
                    f = f_new
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    # Increase sigma on success
                    sigma *= 1.25
                else:
                    # Decrease sigma on failure
                    sigma *= 0.8
                
                if sigma < 1e-10:
                    break
                
                # Check for stagnation
                if sigma < 1e-6 and calls_used < budget:
                    # Try a few random restarts if stuck
                    for _ in range(min(10, budget - calls_used)):
                        if time.time() > deadline:
                            break
                        x_restart = np.clip(np.zeros(n) + rng.standard_normal(n) * 2.0, -5, 5)
                        f_restart = func(x_restart)
                        calls_used += 1
                        if f_restart < best_f:
                            best_f = f_restart
                            best_x = x_restart.copy()
                            x = x_restart
                            f = f_restart
                            sigma = 1.0
                            break
                    if sigma < 1e-6:
                        break
        
        # Phase 3: Local Polish
        # Do a fine grid search around the best point
        if calls_used < budget:
            polish_range = 0.5
            steps = 5
            grid_polish = np.linspace(-polish_range, polish_range, steps)
            
            # Only do this if we have enough budget
            # 5^5 = 3125 is too many. Let's do a coordinate descent polish.
            for d in range(n):
                if calls_used >= budget or time.time() > deadline:
                    break
                # Try small steps in dimension d
                for delta in [-0.1, -0.05, -0.01, 0.01, 0.05, 0.1]:
                    x_try = best_x.copy()
                    x_try[d] += delta
                    x_try = np.clip(x_try, -5, 5)
                    f_try = func(x_try)
                    calls_used += 1
                    if f_try < best_f:
                        best_f = f_try
                        best_x = x_try.copy()
                    
                    if calls_used >= budget or time.time() > deadline:
                        break
            
            # Another pass with smaller steps
            if calls_used < budget:
                for d in range(n):
                    if calls_used >= budget or time.time() > deadline:
                        break
                    for delta in [-0.01, -0.005, -0.001, 0.001, 0.005, 0.01]:
                        x_try = best_x.copy()
                        x_try[d] += delta
                        x_try = np.clip(x_try, -5, 5)
                        f_try = func(x_try)
                        calls_used += 1
                        if f_try < best_f:
                            best_f = f_try
                            best_x = x_try.copy()
                        
                        if calls_used >= budget or time.time() > deadline:
                            break
        
        return best_x, best_f
