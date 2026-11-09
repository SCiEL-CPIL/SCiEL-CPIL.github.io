import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.time_limit = 325.0  # Safety margin for 330s kill

    def __call__(self, func):
        start_time = time.time()
        rng = np.random.default_rng(42)

        n = self.dim
        budget = self.budget

        best_x = np.zeros(n)
        best_f = float('inf')

        # Phase 1: Random initialization to find a good basin
        # Use a small fraction of budget for random sampling
        n_init = min(20, budget // 10)
        for i in range(n_init):
            x = rng.uniform(-5, 5, n)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        budget_used = n_init

        # Phase 2: Nelder-Mead simplex method
        # This exploits local geometry and converges fast in smooth basins
        # We run NM from the best point found in Phase 1
        if budget - budget_used >= 10:
            best_x, best_f = self._nelder_mead(func, best_x, best_f, rng, n, budget - budget_used, start_time)
            budget_used = budget

        # Phase 3: If we have remaining budget (shouldn't happen much), do (1+1)-ES
        if budget_used < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            while budget_used < budget:
                if time.time() - start_time > self.time_limit:
                    break
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                budget_used += 1

        return best_x, best_f

    def _nelder_mead(self, func, x0, f0, rng, n, budget, start_time):
        """Nelder-Mead simplex method with adaptive step size."""
        # Initialize simplex
        x = x0.copy()
        f = f0
        
        # Create initial simplex
        simplex = []
        simplex_f = []
        
        # First point is x0
        simplex.append(x.copy())
        simplex_f.append(f)
        
        # Other points are x0 + small perturbation
        step = 0.5
        for i in range(n):
            xi = x.copy()
            xi[i] += step
            simplex.append(xi.copy())
            simplex_f.append(func(xi))
        
        # Track best
        best_x = x.copy()
        best_f = f
        
        # Nelder-Mead parameters
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrink
        
        budget_used = n + 1
        max_iter = budget - budget_used
        
        for it in range(max_iter):
            if time.time() - start_time > self.time_limit:
                break
            
            # Sort simplex by function value
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            # Update best
            if simplex_f[0] < best_f:
                best_f = simplex_f[0]
                best_x = simplex[0].copy()
            
            # Centroid of all points except worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            worst_f = simplex_f[-1]
            
            # Reflection
            x_r = centroid + alpha * (centroid - worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            budget_used += 1
            
            if f_r < simplex_f[0]:
                # Try expansion
                x_e = centroid + gamma * (centroid - worst)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                budget_used += 1
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # Contraction
                if f_r < worst_f:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (worst - centroid)
                
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                budget_used += 1
                
                if f_c < min(f_r, worst_f):
                    simplex[-1] = x_c
                    simplex_f[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        simplex_f[i] = func(simplex[i])
                        budget_used += 1
                    if budget_used > budget:
                        break
            
            # Check convergence
            spread = np.max(simplex_f) - np.min(simplex_f)
            if spread < 1e-10:
                break
        
        return best_x, best_f
