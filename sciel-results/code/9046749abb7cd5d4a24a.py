import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        
        # Number of restarts
        # More restarts with smaller budgets to explore different regions
        n_restarts = 10
        budget_per_restart = self.budget // n_restarts
        
        # Nelder-Mead parameters
        # Initial step size - smaller for more local search
        delta = 0.8
        
        for r in range(n_restarts):
            # Initialize center
            x0 = rng.uniform(-4, 4, self.dim)
            
            # Initialize simplex with diverse vertices
            simplex = np.zeros((self.dim + 1, self.dim))
            simplex[0] = x0
            for i in range(self.dim):
                simplex[i+1] = x0.copy()
                simplex[i+1][i] += delta
            
            # Evaluate initial simplex
            fvals = np.array([func(v) for v in simplex])
            
            # Track best so far
            idx_min = np.argmin(fvals)
            if fvals[idx_min] < best_f:
                best_f = fvals[idx_min]
                best_x = simplex[idx_min].copy()
            
            calls_used = self.dim + 1
            budget_left = budget_per_restart - calls_used
            
            # Nelder-Mead loop with adaptive parameters
            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction
            sigma = 0.5   # Shrink
            
            # Sort simplex by function value
            def sort_simplex():
                nonlocal simplex, fvals
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]
            
            sort_simplex()
            
            # Track convergence
            prev_spread = np.inf
            
            while budget_left > 0:
                # Current worst point is simplex[-1]
                # Current second worst is simplex[-2]
                # Current best is simplex[0]
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                budget_left -= 1
                
                if f_r < fvals[0]:
                    # Try expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    if budget_left <= 0:
                        break
                    f_e = func(x_e)
                    budget_left -= 1
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (x_r - centroid)
                    
                    x_c = np.clip(x_c, -5, 5)
                    if budget_left <= 0:
                        break
                    f_c = func(x_c)
                    budget_left -= 1
                    
                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if budget_left <= 0:
                                break
                            fvals[i] = func(simplex[i])
                            budget_left -= 1
                
                # Update best
                idx_min = np.argmin(fvals)
                if fvals[idx_min] < best_f:
                    best_f = fvals[idx_min]
                    best_x = simplex[idx_min].copy()
                
                # Sort simplex again for next iteration
                sort_simplex()
                
                # Check for convergence - if spread is very small, break early
                spread = fvals[-1] - fvals[0]
                if spread < 1e-6:
                    break
                if spread > prev_spread * 0.99 and budget_left < 10:
                    break
                prev_spread = spread
        
        return best_x, best_f
