import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = 300.0
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(123)
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        # Multi-start Nelder-Mead
        # We have 500 budget. 
        # Let's use blocks of 100 evaluations for each start.
        num_starts = 5
        
        for start_idx in range(num_starts):
            if time.time() > deadline or budget <= 0:
                break
            
            # Initialize simplex
            # Center at a random point
            center = rng.uniform(-5, 5, dim)
            
            # Simplex points
            simplex = [center.copy()]
            f_values = [func(center)]
            budget -= 1
            if f_values[0] < best_f:
                best_f = f_values[0]
                best_x = center.copy()
            
            # Other points are center + delta * e_i
            delta = 1.0
            for i in range(dim):
                if budget <= 0:
                    break
                p = center.copy()
                p[i] += delta
                p = np.clip(p, -5, 5)
                f_p = func(p)
                simplex.append(p)
                f_values.append(f_p)
                budget -= 1
                
                if f_p < best_f:
                    best_f = f_p
                    best_x = p.copy()
            
            if len(simplex) < dim + 1:
                continue
            
            # Nelder-Mead iterations
            while budget > 0 and time.time() < deadline:
                # Sort simplex by fitness
                indices = np.argsort(f_values)
                simplex = [simplex[i] for i in indices]
                f_values = [f_values[i] for i in indices]
                
                # Check convergence
                if abs(f_values[0] - f_values[-1]) < 1e-10:
                    break
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                
                # Reflection
                alpha = 1.0
                x_r = centroid + alpha * (centroid - worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                budget -= 1
                
                if f_r < f_values[0]:
                    # Expansion
                    gamma = 2.0
                    x_e = centroid + gamma * (centroid - worst)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    budget -= 1
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                elif f_r < f_values[-2]:
                    # Simple Reflection
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                else:
                    # Contraction
                    rho = 0.5
                    x_c = centroid + rho * (worst - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    budget -= 1
                    
                    if f_c < f_values[-1]:
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if budget <= 0:
                                break
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            f_values[i] = func(simplex[i])
                            budget -= 1
                        
                        if budget <= 0:
                            break
                
                # Update best
                if f_values[0] < best_f:
                    best_f = f_values[0]
                    best_x = simplex[0].copy()

        return best_x, best_f
