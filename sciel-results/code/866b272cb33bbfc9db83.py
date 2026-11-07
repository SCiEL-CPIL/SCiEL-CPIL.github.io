"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead Simplex with multi-start and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        rng = np.random.default_rng(42)
        
        # Multi-start: try a few different initializations to avoid bad basins
        # The function is likely unimodal but shifted/rotated, so center is a good bet,
        # but random starts cover the space.
        
        best_x = None
        best_f = np.inf
        
        # We have 500 function calls.
        # Nelder-Mead uses N+1 points initially and updates them.
        # A single run of NM might use 100-300 calls.
        # We can afford 2-3 restarts or one long run with adaptation.
        
        # Let's use a strategy: 
        # 1. Start at origin (or random near origin) with a reasonable step size.
        # 2. Run Nelder-Mead.
        # 3. If budget remains and time allows, try another start or refine.
        
        # Given the "Sharp Ridge" nature, getting close to the ridge is key.
        # NM is good at following ridges once the simplex spans the direction.
        
        def nelder_mead(func, x0, budget_left, start_time, deadline, rng):
            n = self.dim
            # Initialize simplex
            # Start with a small perturbation
            simplex = [x0.copy()]
            f_vals = [func(x0)]
            budget_used = 1
            
            for i in range(n):
                step = 0.5 if i == 0 else 0.1
                xi = x0.copy()
                xi[i] += step
                # Clip to box
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)
                f_vals.append(func(xi))
                budget_used += 1
                
            # Sort simplex by f value
            simplex = np.array(simplex)
            f_vals = np.array(f_vals)
            order = np.argsort(f_vals)
            simplex = simplex[order]
            f_vals = f_vals[order]
            
            best_x_local = simplex[0].copy()
            best_f_local = f_vals[0]
            
            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction
            sigma = 0.5   # Shrink
            
            while budget_used < budget_left:
                if time.time() > deadline - 5:
                    break
                
                # Calculate centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # 1. Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                budget_used += 1
                
                if fr < f_vals[0]:
                    # Try expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    budget_used += 1
                    
                    if fe < fr:
                        simplex[-1] = xe
                        f_vals[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_vals[-1] = fr
                elif fr < f_vals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    f_vals[-1] = fr
                else:
                    # Try contraction
                    # Outside contraction if fr < f_vals[-1]
                    if fr < f_vals[-1]:
                        xc = centroid + rho * (simplex[-1] - centroid)
                    else:
                        xc = centroid - rho * (simplex[-1] - centroid)
                    
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    budget_used += 1
                    
                    if fc <= fr:
                        simplex[-1] = xc
                        f_vals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_vals[i] = func(simplex[i])
                            budget_used += 1
                            if budget_used >= budget_left:
                                break
                        
                        # Re-sort
                        order = np.argsort(f_vals)
                        simplex = simplex[order]
                        f_vals = f_vals[order]
                        break # Restart or exit if shrunk too much? 
                        # Usually shrink implies we're stuck. Let's break to allow restart.
                        
                # Re-sort
                order = np.argsort(f_vals)
                simplex = simplex[order]
                f_vals = f_vals[order]
                
                if simplex[0].copy() is not None:
                    if f_vals[0] < best_f_local:
                        best_f_local = f_vals[0]
                        best_x_local = simplex[0].copy()
                        
                # Convergence check: if simplex is very small, break
                if np.max(np.abs(simplex - centroid)) < 1e-6:
                    break
                    
            return best_x_local, best_f_local, budget_used

        # Strategy: 
        # Use most of the budget for one robust NM run.
        # NM is deterministic given start. 
        # Start at 0 is often good for BBOB if shifts are small, but shifts can be up to 4.
        # Let's do 2 restarts.
        
        starts = [
            np.zeros(self.dim),
            rng.uniform(-2, 2, self.dim)
        ]
        
        total_budget = self.budget
        budget_per_start = total_budget // len(starts)
        
        for i, s in enumerate(starts):
            if time.time() > deadline - 10:
                break
            
            # Adjust budget for last start to use remaining
            b_left = total_budget - sum(1 for _ in range(i)) * budget_per_start # Approx
            # Actually, let's just pass a large number and let time/budget guard inside handle it.
            # But we need to track total calls.
            
            # Let's track calls globally
            if i == 0:
                calls_made = 0
            else:
                calls_made = budget_per_start
            
            # Pass remaining budget
            rem_budget = total_budget - calls_made
            
            x_opt, f_opt, used = nelder_mead(func, s, rem_budget, start_time, deadline, rng)
            
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt
                
            calls_made += used
            
            if time.time() > deadline - 10:
                break
                
        if best_x is None:
             # Fallback
             best_x = np.zeros(self.dim)
             best_f = func(best_x)
             
        return best_x, best_f
# EVOLVE-BLOCK-END
