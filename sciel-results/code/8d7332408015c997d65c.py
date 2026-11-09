import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        n_calls = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')

        def call_f(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget:
                return None
            n_calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def nelder_mead(x0, max_calls, tol=1e-10, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
            """
            Standard Nelder-Mead implementation.
            Returns (best_x, best_f, n_calls_used)
            """
            nonlocal n_calls, best_x, best_f
            
            # Initialize simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            # Use a slightly larger initial step size to handle potential shifts
            step_size = 0.5 
            for i in range(1, dim + 1):
                simplex[i] = x0.copy()
                simplex[i][i - 1] += step_size
            
            # Evaluate initial simplex
            fvals = np.zeros(dim + 1)
            for i in range(dim + 1):
                f = call_f(simplex[i])
                if f is None:
                    return best_x, best_f, n_calls
                fvals[i] = f
            
            # Sort simplex by fvals
            def sort_simplex():
                nonlocal simplex, fvals
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]
            
            sort_simplex()
            
            # Main loop
            while n_calls < budget and n_calls < n_calls + max_calls:
                if time.time() > deadline - 5:
                    break
                
                # Check convergence based on function value spread
                if np.std(fvals) < tol:
                    break

                # Calculate centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = call_f(x_r)
                if f_r is None:
                    break
                
                # If f_r is better than the worst but worse than the best, replace worst
                if fvals[0] <= f_r < fvals[-1]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                    sort_simplex()
                elif f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = call_f(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                    sort_simplex()
                else:
                    # Contraction
                    # Inside contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = call_f(x_c)
                    if f_c is None:
                        break
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                        sort_simplex()
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f = call_f(simplex[i])
                            if f is None:
                                return best_x, best_f, n_calls
                            fvals[i] = f
                        sort_simplex()
                        
            return best_x, best_f, n_calls

        # Multi-start strategy
        # We will run several Nelder-Mead instances.
        # The first one starts from 0.
        # Subsequent ones start from random points or near the current best if it's good.
        
        # Strategy:
        # 1. Run NM from center (0) with a budget of ~150 calls.
        # 2. If stagnation or time, restart from random points.
        # 3. Keep the best point found so far.
        
        # Initial run
        x_start = np.zeros(dim)
        # Give the first run a generous budget, e.g., 150 calls
        max_iter_run = 150
        
        # Run 1
        _, _, _ = nelder_mead(x_start, max_iter_run)
        
        # Check if we should restart
        # If we are stuck (sigma becomes tiny or no improvement), restart.
        # For simplicity, we just do a few random restarts if budget remains.
        
        # Number of restarts is dynamic based on budget used
        # Let's say we allow up to 5 restarts, but each run has a max budget.
        
        restarts = 0
        max_restarts = 5
        calls_per_restart = 60
        
        while n_calls < budget - 10 and restarts < max_restarts and time.time() < deadline - 5:
            # Generate a random start point
            # We can bias towards the current best if it's good, or just random.
            # Random is safer for escaping local minima in Discus if the first run got stuck.
            # However, Discus is unimodal-ish. If the first run converged to the global min, 
            # random restarts will likely find worse points. 
            # But if the first run got stuck in a local min (unlikely for f11 but possible with bad init), 
            # random restart helps.
            
            # Let's try a mix: 50% random, 50% perturbation of best_x
            if rng.random() < 0.5:
                x_start = rng.uniform(-4, 4, dim)
            else:
                x_start = best_x + rng.standard_normal(dim) * 0.5
                x_start = np.clip(x_start, -5, 5)
            
            _, _, _ = nelder_mead(x_start, calls_per_restart)
            restarts += 1
            
            # If we are very close to convergence (best_f is very low relative to initial?), 
            # we might stop early? 
            # For now, just run through the restarts.
            
        # Final Polish with (1+1)-ES on the global best if budget remains
        remaining = budget - n_calls
        if remaining > 10 and time.time() < deadline - 2:
            x = best_x.copy()
            fx = best_f
            sigma_poly = 0.1
            # Use remaining budget for local refinement
            for _ in range(remaining):
                if time.time() > deadline - 1:
                    break
                y = np.clip(x + sigma_poly * rng.standard_normal(dim), -5.0, 5.0)
                fy = call_f(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma_poly *= 1.1
                    if fx < best_f:
                        best_x = x.copy()
                        best_f = fx
                else:
                    sigma_poly *= 0.9
                    sigma_poly = max(sigma_poly, 1e-4)

        return best_x, best_f
