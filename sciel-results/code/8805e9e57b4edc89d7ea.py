import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        n = self.dim
        rng = np.random.default_rng(42)
        
        # Initial point
        x0 = rng.uniform(-5, 5, n)
        
        # Create initial simplex
        # x0 and x0 + step * e_i
        step = 0.5
        simplex = [x0.copy()]
        for i in range(n):
            xi = x0.copy()
            xi[i] += step
            # Clip to box
            xi = np.clip(xi, -5, 5)
            simplex.append(xi)
            
        # Evaluate initial simplex
        fvals = [func(x) for x in simplex]
        evals = len(fvals)
        
        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        best_x = simplex[np.argmin(fvals)]
        best_f = min(fvals)
        
        while evals < self.budget:
            if time.time() > deadline:
                break
                
            # Sort simplex by function value
            # Indices sorted by fvals
            sorted_indices = np.argsort(fvals)
            # Reorder simplex and fvals
            simplex = [simplex[i] for i in sorted_indices]
            fvals = [fvals[i] for i in sorted_indices]
            
            # Worst point is last
            x_worst = simplex[-1]
            f_worst = fvals[-1]
            
            # Centroid of all but worst
            x_centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = x_centroid + alpha * (x_centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            evals += 1
            
            if f_r < fvals[0]:
                # Expansion
                x_e = x_centroid + gamma * (x_r - x_centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                evals += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # Accept reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                if f_r < f_worst:
                    # Outside contraction
                    x_c = x_centroid + rho * (x_r - x_centroid)
                else:
                    # Inside contraction
                    x_c = x_centroid - rho * (x_worst - x_centroid)
                    
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                evals += 1
                
                if f_c < min(f_r, f_worst):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        evals += 1
                        if evals >= self.budget:
                            break
                    if evals >= self.budget:
                        break
                        
            # Update best
            current_best_f = min(fvals)
            if current_best_f < best_f:
                best_f = current_best_f
                best_x = simplex[np.argmin(fvals)]
                
        return best_x, best_f
