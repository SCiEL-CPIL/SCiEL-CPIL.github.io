"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead optimizer with multi-start strategy."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(123)
        
        # Number of restarts
        n_restarts = 2
        budget_per_restart = total_budget // n_restarts
        
        best_x = None
        best_f = float('inf')
        
        for r in range(n_restarts):
            # Initial point
            x0 = rng.uniform(-3, 3, n)
            
            # Initial simplex
            simplex_size = 2.0
            simplex = []
            f_simplex = []
            
            calls_used = 0
            
            # Helper to add point and check budget
            def add_point(x):
                nonlocal calls_used
                if calls_used >= budget_per_restart:
                    return None
                x_clipped = np.clip(x, -5, 5)
                f_val = func(x_clipped)
                calls_used += 1
                return x_clipped, f_val
            
            # Vertex 0: x0
            res = add_point(x0)
            if res is None: continue
            x_curr, f_curr = res
            simplex.append(x_curr)
            f_simplex.append(f_curr)
            
            # Other vertices: x0 + simplex_size * e_i
            for i in range(n):
                x_i = x0.copy()
                x_i[i] += simplex_size
                res = add_point(x_i)
                if res is None:
                    break
                x_i, f_i = res
                simplex.append(x_i)
                f_simplex.append(f_i)
                
            if len(simplex) < 2:
                continue
                
            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            # Main loop
            while calls_used < budget_per_restart:
                # Sort simplex by function value
                idx = np.argsort(f_simplex)
                simplex = [simplex[i] for i in idx]
                f_simplex = [f_simplex[i] for i in idx]
                
                # Check convergence
                if len(f_simplex) > 1:
                    f_range = f_simplex[-1] - f_simplex[0]
                    if f_range < 1e-8:
                        break
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                res = add_point(x_r)
                if res is None: break
                x_r, f_r = res
                
                if f_r < f_simplex[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    res = add_point(x_e)
                    if res is None: break
                    x_e, f_e = res
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    res = add_point(x_c)
                    if res is None: break
                    x_c, f_c = res
                    
                    if f_c < f_simplex[-1]:
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        shrink_done = False
                        for i in range(1, len(simplex)):
                            x_s = simplex[0] + sigma * (simplex[i] - simplex[0])
                            res = add_point(x_s)
                            if res is None:
                                shrink_done = True
                                break
                            x_s, f_s = res
                            simplex[i] = x_s
                            f_simplex[i] = f_s
                        if shrink_done:
                            break
                
                # Update global best
                if f_simplex[0] < best_f:
                    best_f = f_simplex[0]
                    best_x = simplex[0].copy()
                    
        return best_x, best_f
# EVOLVE-BLOCK-END
