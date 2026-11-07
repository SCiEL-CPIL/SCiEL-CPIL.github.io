import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        D = self.dim
        budget = self.budget
        
        # Nelder-Mead Parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrinkage
        
        # Initial Simplex
        # Start at origin, expand along axes with step 2.0
        step = 2.0
        
        simplex = np.zeros((D + 1, D))
        for i in range(D):
            simplex[i + 1, i] = step
            
        # Evaluate initial simplex
        fvals = np.array([func(x) for x in simplex])
        calls_made = D + 1
        
        best_x = simplex[np.argmin(fvals)].copy()
        best_f = np.min(fvals)
        
        # Nelder-Mead Loop
        while calls_made < budget:
            # Sort simplex by function value
            idx = np.argsort(fvals)
            simplex = simplex[idx]
            fvals = fvals[idx]
            
            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()
                
            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # 1. Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            f_r = func(x_r)
            calls_made += 1
            
            if f_r < fvals[0]:
                # 2. Expansion
                x_e = centroid + gamma * (centroid - simplex[-1])
                f_e = func(x_e)
                calls_made += 1
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # Simple reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # 3. Contraction
                if f_r < fvals[-1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (centroid - simplex[-1])
                    
                f_c = func(x_c)
                calls_made += 1
                
                if f_c < min(f_r, fvals[-1]):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # 4. Shrinkage
                    # Only shrink if we have enough budget
                    if calls_made + D >= budget:
                        break
                    for i in range(1, D + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
                        calls_made += 1
                        
                    # Re-evaluate best
                    idx_min = np.argmin(fvals)
                    if fvals[idx_min] < best_f:
                        best_f = fvals[idx_min]
                        best_x = simplex[idx_min].copy()
                        
        return best_x, best_f
