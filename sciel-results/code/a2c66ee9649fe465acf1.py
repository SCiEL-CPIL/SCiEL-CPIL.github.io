import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        # We will use a multi-start Nelder-Mead approach.
        # We split the budget into a few phases.
        # Phase 1: Coarse search with 3 restarts.
        # Phase 2: Fine search on the best result.
        
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)
        
        # Allocate budget: 
        # 3 restarts of ~100 evals each, then 1 final polish of ~100 evals?
        # Total 500.
        # Let's do 2 restarts of 150, and 1 final of 200.
        
        restart_budgets = [130, 130, 220]
        
        for i, r_budget in enumerate(restart_budgets):
            if time.time() > deadline:
                break
                
            # Initialize point
            # The minimizer is in [-4, 4]^5.
            x0 = rng.uniform(-4, 4, self.dim)
            f0 = func(x0)
            
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
                
            # Run Nelder-Mead from x0
            # We need to construct the initial simplex.
            # Standard Nelder-Mead starts with x0 and x0 + delta * e_i
            # This requires d+1 evaluations.
            # We have r_budget left for this phase.
            
            # To save evaluations, we can just run the optimization loop.
            # A standard Nelder-Mead implementation.
            
            x, f = self._nelder_mead(func, x0, f0, r_budget, rng, deadline)
            
            if f < best_f:
                best_f = f
                best_x = x.copy()

        return best_x, best_f

    def _nelder_mead(self, func, x0, f0, budget, rng, deadline):
        # Implement Nelder-Mead
        d = self.dim
        # Initial simplex
        # x0 is already evaluated
        # We need d more points.
        # If budget < d+1, just return x0
        
        if budget < d + 1:
            return x0, f0
            
        simplex = [x0]
        fvals = [f0]
        calls_used = 1
        
        # Generate other vertices
        # Use a small step size for initial simplex
        step = 0.5 
        for i in range(d):
            if calls_used >= budget:
                break
            if time.time() > deadline:
                break
                
            xi = x0.copy()
            xi[i] += step
            # Clip to box? The problem says func clips, but we should stay in bounds for the optimizer's internal logic if possible, or rely on func's clipping.
            # The contract says func clips. So we can pass any vector, it will be clipped.
            # However, it's better to keep the simplex within the box to avoid weird behavior if the function is defined only on the box.
            xi = np.clip(xi, -5, 5)
            
            fi = func(xi)
            calls_used += 1
            simplex.append(xi)
            fvals.append(fi)
            
            if fi < f0:
                f0 = fi
                x0 = xi.copy()
                
        if calls_used >= budget:
            # Return the best point found so far
            idx = np.argmin(fvals)
            return simplex[idx], fvals[idx]

        # Nelder-Mead loop
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrinkage
        
        while calls_used < budget:
            if time.time() > deadline:
                break
                
            # Sort simplex by function value
            # We want the best point first
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Convergence check
            if np.std(fvals) < 1e-12:
                break
                
            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            
            # Reflection
            if calls_used >= budget:
                break
            x_ref = centroid + alpha * (centroid - worst)
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = func(x_ref)
            calls_used += 1
            
            if f_ref < fvals[0]:
                # Expansion
                if calls_used >= budget:
                    break
                x_exp = centroid + gamma * (centroid - worst)
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = func(x_exp)
                calls_used += 1
                
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
            elif f_ref < fvals[-2]:
                # Simple reflection
                simplex[-1] = x_ref
                fvals[-1] = f_ref
            else:
                # Contraction
                # Outside contraction if f_ref < fvals[-1], inside otherwise
                if calls_used >= budget:
                    break
                if f_ref < fvals[-1]:
                    x_con = centroid + rho * (worst - centroid)
                else:
                    x_con = centroid - rho * (worst - centroid)
                    
                x_con = np.clip(x_con, -5, 5)
                f_con = func(x_con)
                calls_used += 1
                
                if f_con < min(f_ref, fvals[-1]):
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrinkage
                    for i in range(1, len(simplex)):
                        if calls_used >= budget:
                            break
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls_used += 1
                    
                    if calls_used >= budget:
                        break

        # Return best point
        idx = np.argmin(fvals)
        return simplex[idx], fvals[idx]
