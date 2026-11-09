import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 330s kill
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def try_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initialization
        # Evaluate origin and a few random points to find good starting locations
        x0 = np.zeros(dim)
        f0 = try_call(x0)
        
        # Evaluate a few random points to identify diverse basins
        n_random = 10
        random_points = []
        for i in range(n_random):
            if calls >= budget:
                break
            xr = rng.uniform(-4, 4, size=dim)
            fr = try_call(xr)
            if fr is not None:
                random_points.append((xr, fr))
        
        # Sort random points by fitness
        random_points.sort(key=lambda x: x[1])
        
        # Select top 3 starting points for multi-start Nelder-Mead
        # Include the origin if it's good, otherwise use top randoms
        starts = []
        if f0 is not None:
            starts.append(x0)
        
        for p in random_points[:2]:
            if p[0] not in starts: # Simple check, though exact match is unlikely
                starts.append(p[0])
        
        # If we have fewer than 3 starts, add more randoms
        while len(starts) < 3 and len(random_points) > len(starts):
            starts.append(random_points[len(starts)][0])

        # Phase 2: Multi-start Nelder-Mead
        # We will allocate the remaining budget among the starts.
        # Since NM is fast, we can run it for a fixed number of iterations or until time runs out.
        # maxiter in scipy's NM corresponds to function evaluations roughly.
        
        remaining_budget = budget - calls
        if remaining_budget > 0 and time.time() < deadline:
            # Allocate budget to starts. 
            # Prioritize the best start.
            # Let's give the best start 50% of the remaining budget, and split the rest among others.
            # Or simpler: run NM on each start with a limited budget, stopping if time is tight.
            
            # To ensure we finish in time, we estimate the cost per NM run.
            # In 5D, NM might take ~20-50 evaluations to converge significantly.
            # Let's try to run 3 instances of NM.
            
            n_starts = len(starts)
            if n_starts == 0:
                return best_x, best_f

            # Dynamically determine maxiter for each NM run
            # We want to use up the budget efficiently.
            # Let's reserve 10 calls for a final polish or safety.
            usable_budget = remaining_budget - 10
            if usable_budget < n_starts * 10: # Minimum 10 calls per start to be useful
                usable_budget = max(0, remaining_budget)
            
            # Split budget among starts
            # Give more budget to the best start
            weights = [1.0] * n_starts
            weights[0] = 2.0 # Best start gets double weight
            total_weight = sum(weights)
            
            for i, start_x in enumerate(starts):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Determine maxiter for this start
                alloc = int(usable_budget * weights[i] / total_weight)
                if alloc < 10:
                    alloc = 10
                # Cap it so we don't exceed remaining budget
                alloc = min(alloc, budget - calls - 5) # Keep 5 calls safety
                
                if alloc <= 0:
                    continue

                try:
                    # Nelder-Mead
                    # xatol and fatol set very small to allow deep optimization
                    result = minimize(
                        func,
                        start_x,
                        method='Nelder-Mead',
                        options={'maxiter': alloc,
                                 'xatol': 1e-12,
                                 'fatol': 1e-15,
                                 'adaptive': True}
                    )
                    # Update best if NM found something better
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5.0, 5.0)
                except Exception:
                    # If NM fails, continue with next start
                    pass

        # Phase 3: Final Polish with (1+1) ES if budget remains
        # This helps if NM stopped early due to maxiter but there is still budget
        if calls < budget and time.time() < deadline and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma_local = 0.5
            
            # Simple adaptive step size
            # Run until budget or time is up
            while calls < budget and time.time() < deadline:
                # Generate a neighbor
                y = x + sigma_local * rng.standard_normal(dim)
                fy = try_call(y)
                if fy is None:
                    break
                
                if fy < fx:
                    x, fx = y, fy
                    # Increase step slightly if we keep succeeding?
                    # Or keep it constant?
                    # Let's use a simple 1/3 rule for step size
                    sigma_local *= 1.1
                else:
                    sigma_local *= 0.8
                
                if sigma_local < 1e-10:
                    break

        return best_x, best_f
