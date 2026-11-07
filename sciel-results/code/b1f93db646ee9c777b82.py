import numpy as np

class Optimizer:
    """Nelder-Mead continuous optimizer for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # NM parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        eps = 1e-8
        
        # Initial simplex
        # Center at 0
        x0 = np.zeros(dim)
        # Initial step size
        delta = 1.0
        
        simplex = [x0.copy()]
        for i in range(dim):
            x = x0.copy()
            x[i] += delta
            simplex.append(x)
            
        # Evaluate initial simplex
        fvals = []
        calls = 0
        for x in simplex:
            if calls >= budget:
                break
            fx = func(x)
            fvals.append(fx)
            calls += 1
            
        if calls < len(simplex):
            # Not enough budget for full simplex, just return best found
            best_x = simplex[np.argmin(fvals)]
            best_fx = min(fvals)
            return best_x, best_fx

        best_x = simplex[0].copy()
        best_fx = float('inf')
        for i in range(len(simplex)):
            if fvals[i] < best_fx:
                best_fx = fvals[i]
                best_x = simplex[i].copy()

        max_iter = 50 # Safety
        stall_count = 0
        stall_limit = 20
        
        while calls < budget:
            # Sort simplex
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Update global best
            if fvals[0] < best_fx:
                best_fx = fvals[0]
                best_x = simplex[0].copy()
                stall_count = 0
            else:
                stall_count += 1
                
            if stall_count > stall_limit:
                # Restart: move best point slightly and rebuild simplex
                # Or just perturb best point and continue
                # Let's perturb best point and reset simplex around it
                best_x = best_x + rng.normal(0, 0.1, dim)
                best_x = np.clip(best_x, -5, 5)
                simplex = [best_x.copy()]
                for i in range(dim):
                    x = best_x.copy()
                    x[i] += delta
                    simplex.append(x)
                fvals = []
                for x in simplex:
                    if calls >= budget:
                        break
                    fx = func(x)
                    fvals.append(fx)
                    calls += 1
                if calls < len(simplex):
                    break
                continue

            # Centroid of all but worst
            x_worst = simplex[-1]
            f_worst = fvals[-1]
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_ref = centroid + alpha * (centroid - x_worst)
            x_ref = np.clip(x_ref, -5, 5)
            
            if calls >= budget: break
            f_ref = func(x_ref)
            calls += 1
            
            if f_ref < fvals[0]:
                # Expansion
                if calls >= budget: break
                x_exp = centroid + gamma * (x_ref - x_worst)
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = func(x_exp)
                calls += 1
                
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
                # Inside contraction
                x_con = centroid - rho * (x_worst - centroid)
                x_con = np.clip(x_con, -5, 5)
                
                if calls >= budget: break
                f_con = func(x_con)
                calls += 1
                
                if f_con < f_worst:
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        if calls >= budget: break
                        fvals[i] = func(simplex[i])
                        calls += 1
                    
                    # Check if we ran out of budget during shrink
                    if calls >= budget:
                         # We might have updated some fvals, need to handle carefully
                         # But the loop will break anyway
                         pass

        return best_x, best_fx
