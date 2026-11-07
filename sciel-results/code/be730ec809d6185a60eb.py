import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)

    def __call__(self, func):
        t_start = time.time()
        calls = 0
        max_calls = self.budget
        
        # Helper to call func safely
        def safe_call(x):
            nonlocal calls
            if calls >= max_calls:
                return None
            if time.time() - t_start > 280:
                return None
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = func(x_clipped)
            calls += 1
            return fx

        # 1. Global Sampling
        n_global = max(10, self.budget // 10) # 10% for global
        points = []
        values = []
        
        # Random sampling
        for _ in range(n_global):
            x = np.random.uniform(-5.0, 5.0, self.dim)
            fx = safe_call(x)
            if fx is None:
                break
            points.append(x)
            values.append(fx)
            
        if not points:
            return None # Should not happen
            
        # Sort by value
        sorted_idx = np.argsort(values)
        points = [points[i] for i in sorted_idx]
        values = [values[i] for i in sorted_idx]
        
        x_best = points[0]
        f_best = values[0]
        
        # 2. Local Optimization with Nelder-Mead
        # We will try to optimize starting from the best points found
        # We have budget - n_global calls left
        
        remaining_budget = self.budget - calls
        
        # Simplex initialization parameters
        initial_step = 1.0 
        
        for i in range(len(points)):
            if calls >= max_calls:
                break
            if time.time() - t_start > 280:
                break
                
            x0 = points[i]
            f0 = values[i]
            
            # Build initial simplex
            simplex = [x0.copy()]
            simplex_f = [f0]
            
            for d in range(self.dim):
                x_plus = x0.copy()
                x_plus[d] += initial_step
                x_minus = x0.copy()
                x_minus[d] -= initial_step
                
                fx_plus = safe_call(x_plus)
                if fx_plus is None: break
                simplex.append(x_plus)
                simplex_f.append(fx_plus)
                
                # Note: We don't need x_minus for standard NM, just +step for initial simplex
                # Standard NM simplex has D+1 points. We have 1 (center) + D (axes) = D+1.
                
            if len(simplex) < self.dim + 1:
                continue
                
            # Run Nelder-Mead
            # We need to manage the simplex list
            # To save code size, we implement a compact NM loop
            
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            max_nm_iterations = 100 # Safety cap per restart
            
            for _ in range(max_nm_iterations):
                if calls >= max_calls: break
                if time.time() - t_start > 280: break
                
                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                x_worst = simplex[-1]
                f_worst = simplex_f[-1]
                x_best_nm = simplex[0]
                f_best_nm = simplex_f[0]
                
                # Check convergence
                if f_worst - f_best_nm < 1e-8:
                    break
                # Check simplex size
                spread = np.max(np.abs(np.array(simplex) - x_best_nm))
                if spread < 1e-6:
                    break
                    
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - x_worst)
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = safe_call(x_r)
                if f_r is None: break
                
                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = safe_call(x_e)
                    if f_e is None: break
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    # Just replace worst
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (x_worst - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = safe_call(x_c)
                    if f_c is None: break
                    
                    if f_c < f_worst:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for j in range(1, len(simplex)):
                            simplex[j] = x_best_nm + sigma * (simplex[j] - x_best_nm)
                            simplex[j] = np.clip(simplex[j], -5.0, 5.0)
                            f_s = safe_call(simplex[j])
                            if f_s is None:
                                # If we run out of budget/time during shrink, just break
                                break
                            simplex_f[j] = f_s
                        # If we broke out of loop early, break outer too
                        if calls >= max_calls or time.time() - t_start > 280:
                            break
                            
                # Update global best
                current_best_in_simplex = min(simplex_f)
                if current_best_in_simplex < f_best:
                    f_best = current_best_in_simplex
                    x_best = simplex[np.argmin(simplex_f)]
                    
            # If we used up budget, stop
            if calls >= max_calls:
                break

        return x_best, f_best
