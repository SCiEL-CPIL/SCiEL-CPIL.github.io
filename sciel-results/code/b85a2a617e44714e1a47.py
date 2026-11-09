import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0 # Safety margin
        
        # State
        best_x = None
        best_f = float('inf')
        calls = 0
        
        def safe_func(x):
            nonlocal calls
            if time.time() > deadline:
                raise TimeoutError()
            calls += 1
            return func(x)

        # Phase 1: Random Sampling
        # Use 10% of budget for sampling, min 20, max 100
        sample_budget = min(50, self.budget // 5)
        
        for _ in range(sample_budget):
            if time.time() > deadline: break
            x = np.random.uniform(-5, 5, self.dim)
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = safe_func(best_x)

        # Phase 2: Nelder-Mead
        # Remaining budget
        remaining = self.budget - calls
        
        # Initialize Simplex
        # Center at best_x
        x0 = best_x
        sigma = 0.5 # Initial step size
        
        simplex = [x0.copy()]
        for i in range(self.dim):
            xi = x0.copy()
            # Perturb one dimension
            # If at boundary, perturb inwards? 
            # func clips, so perturbing outwards is fine, it just stays at boundary.
            # But for NM to work, we need distinct points.
            # If x0 is at 5, xi[i] = 5 + sigma. func(xi) = func(5).
            # This might cause duplicate points in simplex if many are at boundary.
            # To avoid this, we can adjust perturbation direction based on position.
            if xi[i] + sigma > 5:
                xi[i] -= 2 * sigma
            elif xi[i] - sigma < -5:
                xi[i] += 2 * sigma
            else:
                xi[i] += sigma
            simplex.append(xi)
            
        # Evaluate Simplex
        simplex_f = []
        for i in range(self.dim + 1):
            if time.time() > deadline: break
            simplex_f.append(safe_func(simplex[i]))
            
        # Sort simplex by function value
        # We need to keep track of indices or just sort the list of tuples
        # Let's store as list of (f, x)
        simplex_data = list(zip(simplex_f, simplex))
        
        # Main NM Loop
        alpha = 1.0 # Reflection
        gamma = 2.0 # Expansion
        rho = 0.5   # Contraction
        sigma_shrink = 0.5 # Shrink
        
        while calls < self.budget - 5: # Keep 5 calls buffer
            if time.time() > deadline:
                break
                
            # Sort
            simplex_data.sort(key=lambda item: item[0])
            f_worst = simplex_data[-1][0]
            x_worst = simplex_data[-1][1]
            f_best = simplex_data[0][0]
            x_best = simplex_data[0][1]
            
            # Update global best
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()
                
            # Convergence check
            # If spread is small, stop
            f_spread = f_worst - f_best
            if f_spread < 1e-8:
                break
                
            # Centroid of all but worst
            centroid = np.mean([item[1] for item in simplex_data[:-1]], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - x_worst)
            f_r = safe_func(x_r)
            
            if f_r < f_best:
                # Expansion
                x_e = centroid + gamma * (centroid - x_worst)
                f_e = safe_func(x_e)
                if f_e < f_r:
                    simplex_data[-1] = (f_e, x_e)
                else:
                    simplex_data[-1] = (f_r, x_r)
            elif f_r < simplex_data[-2][0]:
                # Reflection is good but not best
                simplex_data[-1] = (f_r, x_r)
            else:
                # Contraction
                # Outer contraction (if f_r < f_worst)
                if f_r < f_worst:
                    x_c = centroid + rho * (x_r - centroid)
                    f_c = safe_func(x_c)
                    if f_c <= f_r:
                        simplex_data[-1] = (f_c, x_c)
                    else:
                        # Shrink
                        for i in range(1, len(simplex_data)):
                            x_s = x_best + sigma_shrink * (simplex_data[i][1] - x_best)
                            f_s = safe_func(x_s)
                            simplex_data[i] = (f_s, x_s)
                else:
                    # Inner contraction
                    x_c = centroid - rho * (centroid - x_worst)
                    f_c = safe_func(x_c)
                    if f_c < f_worst:
                        simplex_data[-1] = (f_c, x_c)
                    else:
                        # Shrink
                        for i in range(1, len(simplex_data)):
                            x_s = x_best + sigma_shrink * (simplex_data[i][1] - x_best)
                            f_s = safe_func(x_s)
                            simplex_data[i] = (f_s, x_s)
                            
        return best_x, best_f
