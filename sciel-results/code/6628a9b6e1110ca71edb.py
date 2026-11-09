import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 300s limit, use 280s to be safe
        deadline = start_time + 280
        
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)
        
        # Strategy: Multi-start Nelder-Mead
        # Split budget into 3 phases
        # Phase 1: 150 evals
        # Phase 2: 150 evals
        # Phase 3: 200 evals (polish)
        
        # If budget is small, adjust
        if self.budget < 50:
            # Just do a random search or single NM
            x0 = rng.uniform(-4, 4, self.dim)
            f0 = func(x0)
            return x0, f0

        restart_budgets = [150, 150, 200]
        
        # Ensure sum of restart_budgets <= budget
        # If budget is larger, we could add more, but 500 is fixed.
        # If budget is smaller, we scale down.
        total_restarts = sum(restart_budgets)
        if total_restarts > self.budget:
            scale = self.budget / total_restarts
            restart_budgets = [int(b * scale) for b in restart_budgets]
            # Adjust last one to match exactly
            restart_budgets[-1] += self.budget - sum(restart_budgets)

        for i, r_budget in enumerate(restart_budgets):
            if time.time() > deadline:
                break
            if r_budget <= 0:
                continue
                
            # Initialize point
            # Minimizer is in [-4, 4]^5
            x0 = rng.uniform(-4, 4, self.dim)
            f0 = func(x0)
            
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
                
            # Run Nelder-Mead from x0
            x, f = self._nelder_mead(func, x0, f0, r_budget, rng, deadline)
            
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if best_x is None:
            # Fallback
            x0 = np.zeros(self.dim)
            f0 = func(x0)
            best_x = x0
            best_f = f0

        return best_x, best_f

    def _nelder_mead(self, func, x0, f0, budget, rng, deadline):
        d = self.dim
        if budget < d + 1:
            return x0, f0
            
        simplex = [x0.copy()]
        fvals = [f0]
        calls_used = 1
        
        # Generate other vertices
        # Use a step size relative to the box size
        # Box is [-5, 5], width 10.
        # Start with 0.5
        step = 0.5 
        
        for i in range(d):
            if time.time() > deadline or calls_used >= budget:
                break
                
            xi = x0.copy()
            xi[i] += step
            xi = np.clip(xi, -5, 5)
            
            fi = func(xi)
            calls_used += 1
            simplex.append(xi)
            fvals.append(fi)
            
            if fi < f0:
                f0 = fi
                x0 = xi.copy()
                
        if calls_used >= budget:
            idx = np.argmin(fvals)
            return simplex[idx], fvals[idx]

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Main loop
        while calls_used < budget:
            if time.time() > deadline:
                break
                
            # Sort
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Convergence
            if np.std(fvals) < 1e-12:
                break
                
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            
            # Reflection
            x_ref = centroid + alpha * (centroid - worst)
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = func(x_ref)
            calls_used += 1
            
            if f_ref < fvals[0]:
                # Expansion
                if calls_used >= budget:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
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
                simplex[-1] = x_ref
                fvals[-1] = f_ref
            else:
                # Contraction
                if f_ref < fvals[-1]:
                    x_con = centroid + rho * (worst - centroid)
                else:
                    x_con = centroid - rho * (worst - centroid)
                    
                x_con = np.clip(x_con, -5, 5)
                
                if calls_used >= budget:
                    break
                    
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

        idx = np.argmin(fvals)
        return simplex[idx], fvals[idx]
